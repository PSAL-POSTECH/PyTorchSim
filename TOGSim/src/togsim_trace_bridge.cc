// togsim_trace_bridge.cc -- see togsim_trace_bridge.h
#include "togsim_trace_bridge.h"

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "Tile.h"
#include "Instruction.h"

namespace {

// `uniq` is a per-DMA-record Core tag key, so every reduction iteration of one
// static dma gets a distinct key (multi-tile-K, conv); its memory_barrier reuses
// it. `tag_idx` (the subtile slot) still drives the SRAM double-buffer model.

// FIXME: `uniq` is reconstructed here from record order. build_skeleton should
// instead thread dma_fine_grained's per-iteration tag alloc through as an SSA
// handle on togsim.dma / togsim.memory_barrier (sec 11).
std::shared_ptr<Instruction> make_dma(const togsim::TraceRec& t, int64_t uniq) {
  Opcode op = (t.dir == 1) ? Opcode::MOVOUT : Opcode::MOVIN;
  std::vector<size_t> tile_size(t.dims.begin(), t.dims.end());
  std::vector<int> tile_stride(t.strides.begin(), t.strides.end());
  std::vector<int64_t> tag_idx{(int64_t)t.tag_slot};
  std::vector<int64_t> tag_stride{1};
  auto inst = std::make_shared<Instruction>(
      op, /*compute_cycle=*/0, /*num_parents=*/0, /*dram_addr=*/t.addr,
      tile_size, tile_stride, (size_t)t.elem_bits, tag_idx, tag_stride,
      /*accum_tag_idx_list=*/std::vector<int64_t>{});
  inst->set_is_async(t.is_async != 0);
  inst->set_addr_name("tag" + std::to_string(uniq), uniq);
  inst->prepare_tag_key();
  return inst;
}

// A MEMORY_BAR carrying the SAME `uniq` tag key as the async dma it gates -- the
// Core's tag table signals it at the dma's DATA-ready (resp-complete), unlike a
// raw DONE edge that the async dma releases at issue-complete.
std::shared_ptr<Instruction> make_mem_bar(const togsim::TraceRec& t, int64_t uniq) {
  auto bar = std::make_shared<Instruction>(
      Opcode::MEMORY_BAR, 0, 0, 0,
      std::vector<size_t>{}, std::vector<int>{}, 0,
      std::vector<int64_t>{(int64_t)t.tag_slot}, std::vector<int64_t>{1},
      std::vector<int64_t>{});
  bar->set_addr_name("tag" + std::to_string(uniq), uniq);
  bar->prepare_tag_key();
  return bar;
}

std::shared_ptr<Instruction> make_compute(const togsim::TraceRec& t) {
  auto inst = std::make_shared<Instruction>(
      Opcode::COMP, /*compute_cycle=*/(cycle_type)t.cycle, /*num_parents=*/0,
      /*dram_addr=*/0, std::vector<size_t>{}, std::vector<int>{}, /*elem_bits=*/0,
      std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
  inst->set_overlapping_cycle((cycle_type)t.overlapping);
  inst->set_compute_type(t.compute_type);  // route to VPU vs systolic array
  return inst;
}

}  // namespace

std::unique_ptr<TileGraph> trace_to_tilegraph(
    const char* so_path, const int64_t* shape_args, int32_t n_shape,
    const uint64_t* tensor_base, int32_t n_tensors,
    const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
    const int32_t* partition_cores, int32_t n_partition_cores,
    const std::string& name) {
  using togsim::TraceRec;
  // The record stream is never materialized: each pass replays the producer and
  // consumes records as they are emitted. togsim_kernel is a pure emitter, so the
  // replay is identical; retaining the stream would double peak memory (a large
  // 8x8 conv OOMs). See togsim_loader.h run_producer_stream.
  auto run_pass = [&](const togsim::TraceSink& sink) {
    return togsim::run_producer_stream(so_path, shape_args, n_shape,
                                       tensor_base, n_tensors, cyc, ovl, n_tiles,
                                       partition_cores, n_partition_cores, sink);
  };

  // Spad bytes per buffer id, taken from the DMA records that touch it (load fills
  // its dst, store drains its src) -- the authoritative tile size. A compute output
  // (never DMA-loaded but stored) gets its footprint from its store record. Built
  // in a pre-pass so it is known before the producing compute is processed.
  auto rec_bytes = [](const TraceRec& t) {        // single source of the tile footprint
    size_t numel = 1;
    for (auto d : t.dims) numel *= (size_t)d;
    return numel * (t.elem_bits / 8);
  };
  std::map<int64_t, size_t> buf_bytes;
  if (!run_pass([&](const TraceRec& t) {          // PASS 1 -- footprints only
        if (t.kind != TraceRec::DMA) return;
        const auto& bs = (t.dir == 1) ? t.read_bufs : t.write_bufs;  // store reads spad, load writes spad
        for (int64_t b : bs) buf_bytes[b] = rec_bytes(t);
      }))
    return nullptr;

  auto tg = std::make_unique<TileGraph>(name, name);
  // Empty cache plan (no L2/CMEM persistence) -- append_subgraph propagates it
  // to each subgraph, and DMA::is_cacheable dereferences it, so it must be a
  // valid (if empty) IntervalTree rather than null.
  tg->init_cache_plan({});

  std::shared_ptr<TileSubGraph> sg;
  std::shared_ptr<Tile> tile;
  // The explicit dependency DAG (sec 10): per SRAM buffer, writers(b) is the SET of
  // current producers' DONE-handles and readers(b) its consumers. Scoped per
  // work-item, so distinct work-items are independent (-> parallel).
  std::map<int64_t, std::vector<std::shared_ptr<Instruction>>> writers;       // buffer id -> current producers (DONE-handles)
  // The non-MATMUL subset of writers(b) (the accumulator's init/bias seed). An
  // is_mm_accum UNION only ever appends MATMULs, so this set is fixed between
  // REPLACEs -- keeping it lets the UNION skip re-scanning the whole (O(K)-long)
  // writers(b) on every one of the K accumulating matmuls, which was O(K^2).
  std::map<int64_t, std::vector<std::shared_ptr<Instruction>>> seeds;
  // An async dma is paired with its explicit memory_barrier(s) by the runtime tag
  // (tag_id, tag_slot). It is 1 load : N barriers (the load happens once per
  // reduction iteration; each consumer in that iteration is preceded by a wait on
  // the same tag), so we track the CURRENT (most recent) load per (tag_id,
  // tag_slot) -- not a FIFO. Each load gets a fresh `uniq` Core key, so successive
  // reduction iterations (multi-tile-K, conv) never collide in the tag table; the
  // iteration's barriers reuse that load's uniq. Correct because the load nest and
  // its consumer nest run in order within the reduction body (no cross-iteration
  // prefetch). Scoped per work-item.
  std::map<std::pair<int32_t, uint64_t>,
           std::pair<int64_t, std::shared_ptr<Instruction>>> current_dma;
  // Dedup barriers on the CURRENT load of a (tag_id, tag_slot): a conv reads one
  // loaded subtile from many matmuls, so its per-consumer waits collapse to one bar.
  // A new load bumps uniq, so a genuine new wait still gets its own bar.
  std::map<std::pair<int32_t, uint64_t>,
           std::pair<int64_t, std::shared_ptr<Instruction>>> bar_for_load;
  int64_t next_tag = 0;   // mints a unique Core tag key per dma record
  int cur_tile_group = -1;   // work-item index, bumped per TILE_BEGIN (trace grouping)
  std::set<int64_t> cur_tile_bufs;   // distinct spad buffers this tile touches
  size_t cur_tile_footprint = 0;     // their footprint sum = the tile's resident set

  auto flush = [&]() {
    if (sg && tile) {
      tile->set_spad_footprint(cur_tile_footprint);   // distinct-buffer resident set (1- vs 2-dispatch)
      sg->add_tile(tile);
      tile->set_owner(sg);
      tg->append_subgraph(sg);
    }
    sg.reset();
    tile.reset();
    writers.clear();
    seeds.clear();
    current_dma.clear();
    bar_for_load.clear();
    cur_tile_bufs.clear();
    cur_tile_footprint = 0;
    next_tag = 0;
  };

  // The single dataflow rule (sec 10.3). READ b: depend on all writers(b), ISSUE
  // when both are SA ops else DONE. WRITE b: replace writers(b), except a commutative
  // matmul accumulator, which waits only the seed and unions. WAR: resource models.
  const int MATMUL_CT = 1, PRELOAD_CT = 2;
  auto is_mm_accum = [&](const std::shared_ptr<Instruction>& inst, int64_t b,
                         const std::vector<int64_t>& writes) {
    if (inst->get_compute_type() != MATMUL_CT) return false;
    for (int64_t w : writes) if (w == b) return true;
    return false;
  };
  // REPLACE writers(b) and recompute its seed set in one place.
  auto set_writer = [&](int64_t b, const std::shared_ptr<Instruction>& w) {
    writers[b] = { w };
    auto& sd = seeds[b];
    sd.clear();
    if (w->get_compute_type() != MATMUL_CT) sd.push_back(w);
  };
  auto link = [&](std::shared_ptr<Instruction> inst,
                  const std::vector<int64_t>& reads,
                  const std::vector<int64_t>& writes) {
    for (int64_t b : reads) {
      if (is_mm_accum(inst, b, writes)) continue;   // accumulator read -> handled in WRITE (UNION)
      auto it = writers.find(b);
      if (it != writers.end())
        for (auto& w : it->second) {
          int pct = w->get_compute_type();
          // both SA ops -> occupancy (overlap on the SA pipeline); else latency.
          DepEvent on = (inst->get_compute_type() == MATMUL_CT &&
                         (pct == MATMUL_CT || pct == PRELOAD_CT))
                            ? DepEvent::ISSUE : DepEvent::DONE;
          w->add_dep(inst, on);
        }
    }
    for (int64_t b : writes) {
      if (is_mm_accum(inst, b, writes)) {            // UNION (commutative accumulate)
        auto it = seeds.find(b);
        if (it != seeds.end())
          for (auto& s : it->second)
            s->add_dep(inst, DepEvent::DONE);   // wait the init/bias seed only
        writers[b].push_back(inst);        // join; no reset, no co-matmul edge
      } else {                             // REPLACE (normal output; resets the producer set)
        set_writer(b, inst);
      }
    }
    tile->append_instuction(inst);
  };

  // SRAM-capacity tracking (sec 10.4): a coarse tile is one version of its buffer,
  // freed once all its consumers have issued. NOT reset in flush() -- the spad is a
  // physical per-core resource. v1 is single-core; multi-core would key by (core, buf).
  int64_t next_alloc = 0;
  std::map<int64_t, int64_t> cur_alloc;   // buf -> current version id
  std::map<int64_t, bool> open_ver;       // buf -> version still accepting writes
  struct Ver { std::vector<std::shared_ptr<Instruction>> loads, readers; };
  std::map<int64_t, Ver> vers;
  // Add each buffer once to the current tile's footprint (reloads in a K-loop reuse the same id).
  auto note_bufs = [&](const std::vector<int64_t>& bufs) {
    for (int64_t b : bufs)
      if (cur_tile_bufs.insert(b).second) {
        auto it = buf_bytes.find(b);
        if (it != buf_bytes.end()) cur_tile_footprint += it->second;
      }
  };
  auto sram_on_load = [&](int64_t b, const std::shared_ptr<Instruction>& ld) {
    if (!cur_alloc.count(b) || !open_ver[b]) {   // a read closed it -> new version
      cur_alloc[b] = next_alloc++;
      open_ver[b] = true;
      vers[cur_alloc[b]] = {};
    }
    ld->set_sram_alloc(cur_alloc[b]);
    vers[cur_alloc[b]].loads.push_back(ld);
  };
  // A compute that freshly produces b opens a version like a load, carrying b's
  // footprint; its last reader frees it -- identical lifecycle to a load.
  auto sram_on_write = [&](int64_t b, const std::shared_ptr<Instruction>& w) {
    auto bb = buf_bytes.find(b);
    if (bb == buf_bytes.end()) return;           // size unknown (never DMA'd) -> untracked
    if (!cur_alloc.count(b) || !open_ver[b]) {   // a consuming read closed it -> new version
      cur_alloc[b] = next_alloc++;
      open_ver[b] = true;
      vers[cur_alloc[b]] = {};
      w->set_sram_alloc(cur_alloc[b]);
      w->set_sram_footprint(bb->second);
      vers[cur_alloc[b]].loads.push_back(w);
    }
    // already-open version (further producing writes): same physical bytes, no re-add.
  };
  auto sram_on_read = [&](int64_t b, const std::shared_ptr<Instruction>& rd) {
    auto it = cur_alloc.find(b);
    if (it == cur_alloc.end()) return;           // not a load buffer -> untracked
    vers[it->second].readers.push_back(rd);
    open_ver[b] = false;                          // next write starts a new version
  };
  auto sram_finalize = [&]() {                    // tag only each version's LAST reader
    for (auto& kv : vers) {
      auto& v = kv.second;
      if (v.readers.empty()) {                    // no consumer -> never freed: untrack
        for (auto& ld : v.loads) ld->set_sram_alloc(-1);
        continue;
      }
      v.readers.back()->add_sram_release(kv.first);  // it frees the whole version on issue
    }
  };

  auto feed = [&](const TraceRec& t) {            // PASS 2 -- build, one record at a time
    if (t.kind == TraceRec::TILE_BEGIN) {
      // togsim_dispatch opened a work-item -> new subgraph (bound to its core) +
      // tile. The scope runs until the matching TILE_END (the dispatch wrapper
      // brackets the tile fn call), not until the next begin.
      flush();
      sg = std::make_shared<TileSubGraph>();
      sg->set_core_id(t.core);
      tile = std::make_shared<Tile>(Tile::Status::INITIALIZED);
      cur_tile_group++;
      return;
    }
    if (t.kind == TraceRec::TILE_END) {
      flush();   // close the work-item explicitly (scope = the tile fn call)
      return;
    }
    if (!tile) return;  // defensive: ops before the first TILE_BEGIN

    if (t.kind == TraceRec::DMA) {
      int64_t uniq = next_tag++;                         // fresh Core tag key per dma record
      auto inst = make_dma(t, uniq);
      inst->set_tile_group(cur_tile_group);
      tile->inc_required_sram_size(rec_bytes(t));         // SRAM footprint (ready-tile ordering)
      note_bufs(t.read_bufs); note_bufs(t.write_bufs);   // distinct-buffer footprint for 1- vs 2-dispatch
      if (t.dir == 1) {                                  // STORE
        // store reads the result buffer(s) -> link() JOINs all their writers.
        link(inst, t.read_bufs, t.write_bufs);
        for (int64_t b : t.read_bufs) sram_on_read(b, inst);  // store frees what it drains
      } else {                                           // LOAD
        tile->append_instuction(inst);
        // async load: the CURRENT load for this (tag_id, tag_slot), with a fresh
        // uniq its barriers reuse. writers = the dma until its barrier overwrites it,
        // so consumers gate on arrival. A sync load blocks to arrival itself.
        if (t.is_async) current_dma[{t.tag_id, t.tag_slot}] = {uniq, inst};
        for (int64_t b : t.write_bufs) {
          // No hard WAR edge here: load-buffer reuse (double-buffering, X_spad/
          // W_spad reloaded each reduction iter) is modeled by the SRAM
          // version/capacity machinery (sram_on_load), which sizes how many
          // versions physically coexist. A latency WAR edge would force
          // single-buffering and kill the overlap the spad permits. (The
          // accumulator Y is NOT a load buffer -> its cross-tile WAR is handled by
          // the REPLACE branch of link() when the next tile's init overwrites it.)
          set_writer(b, inst);
          sram_on_load(b, inst);                         // occupy spad
        }
      }
    } else if (t.kind == TraceRec::MEMORY_BAR) {
      // The explicit async-DMA sync. Pair with the CURRENT load for this (tag_id,
      // tag_slot), reusing its uniq: the dma releases the bar at issue, the bar parks
      // on the tag until resp-complete, and becomes the load's handle in writers(b).
      auto it = current_dma.find({t.tag_id, t.tag_slot});
      int64_t uniq = next_tag++;                         // fallback if unpaired
      std::shared_ptr<Instruction> dma_inst;
      if (it != current_dma.end()) { uniq = it->second.first; dma_inst = it->second.second; }
      // Identical wait (same slot, same load instance) already has a barrier -> reuse it
      // so the buffer's consumers gate on it, instead of emitting a redundant barrier.
      auto bf = bar_for_load.find({t.tag_id, t.tag_slot});
      if (bf != bar_for_load.end() && bf->second.first == uniq) {
        for (int64_t b : t.write_bufs) set_writer(b, bf->second.second);
        return;
      }
      auto bar = make_mem_bar(t, uniq);
      bar->set_tile_group(cur_tile_group);
      if (dma_inst) dma_inst->add_dep(bar, DepEvent::DONE);
      tile->append_instuction(bar);
      // the bar is the load's DONE-handle: REPLACE writers(b) with it (no WAR -- the
      // load already WAR'd the prior readers when it wrote).
      for (int64_t b : t.write_bufs) set_writer(b, bar);
      bar_for_load[{t.tag_id, t.tag_slot}] = {uniq, bar};
    } else if (t.kind == TraceRec::COMPUTE) {
      auto inst = make_compute(t);
      inst->set_tile_group(cur_tile_group);
      link(inst, t.read_bufs, t.write_bufs);
      note_bufs(t.read_bufs); note_bufs(t.write_bufs);   // distinct-buffer footprint for 1- vs 2-dispatch
      // in-place buffers (read AND written) are version-transparent (accumulator,
      // in-place vector): skip the self-read and the self-write so footprint is not
      // double-counted. read_bufs/write_bufs are tiny, so a linear scan beats a set.
      auto in = [](const std::vector<int64_t>& v, int64_t b) {
        return std::find(v.begin(), v.end(), b) != v.end();
      };
      for (int64_t b : t.read_bufs)  if (!in(t.write_bufs, b)) sram_on_read(b, inst);   // consuming reads
      for (int64_t b : t.write_bufs) if (!in(t.read_bufs, b))  sram_on_write(b, inst);  // fresh outputs
    }
  };
  if (!run_pass(feed)) return nullptr;
  flush();
  sram_finalize();   // readers per version are now final -> set each version's refcount
  return tg;
}
