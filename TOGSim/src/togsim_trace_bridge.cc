// togsim_trace_bridge.cc -- see togsim_trace_bridge.h
#include "togsim_trace_bridge.h"

#include <algorithm>
#include <map>
#include <set>
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


// All builder state for one kernel. Heap-allocated and owned by the TileGraph's
// tile source, so it survives between on-demand materializations of a tile.
struct BuildState {
  togsim::LazyProducer prod;
  size_t next = 0;               // next work-item to materialize
  std::map<int64_t, size_t> buf_bytes;
  std::shared_ptr<TileSubGraph> sg_out;   // the tile just built

  std::shared_ptr<TileSubGraph> sg;
  std::shared_ptr<Tile> tile;
  std::map<int64_t, std::vector<std::shared_ptr<Instruction>>> writers;
  std::map<int64_t, std::vector<std::shared_ptr<Instruction>>> seeds;
  std::map<std::pair<int32_t, uint64_t>,
           std::pair<int64_t, std::shared_ptr<Instruction>>> current_dma;
  std::map<std::pair<int32_t, uint64_t>,
           std::pair<int64_t, std::shared_ptr<Instruction>>> bar_for_load;
  int64_t next_tag = 0;
  int cur_tile_group = -1;
  std::set<int64_t> cur_tile_bufs;
  size_t cur_tile_footprint = 0;

  // SRAM buffer versions. A version is NOT scoped to a work-item -- the spad is
  // one physical resource, so a buffer reused by the next tile is a new version
  // that must wait for the old one to free. The version *schedule* is therefore
  // precomputed by sram_schedule(), which allocates nothing: the builder can tag
  // a version's last reader as it goes instead of retaining every reader until
  // end-of-stream (which is what forced the whole graph to stay in memory).
  int64_t next_alloc = 0;
  std::map<int64_t, int64_t> cur_alloc;   // buf -> current version id
  std::map<int64_t, bool> open_ver;       // buf -> version still accepting writes
  std::vector<char> has_readers;                        // version -> ever read?
  std::vector<std::pair<size_t, size_t>> last_reader;   // version -> (work-item, record)
  size_t item = 0, rec = 0;               // position of the record being fed
};

// Walk the record stream once, allocating nothing, and work out for each SRAM
// buffer version (a) whether anything ever reads it and (b) where its LAST
// reader sits. Mirrors sram_on_load / sram_on_write / sram_on_read exactly, so
// the version ids it mints are the ones the builder will mint.
void sram_schedule(BuildState& S) {
  using togsim::TraceRec;
  int64_t next_alloc = 0;
  std::map<int64_t, int64_t> cur_alloc;
  std::map<int64_t, bool> open_ver;
  size_t item = 0, rec = 0, pos = 0;
  auto on_open = [&](int64_t b) {
    if (!cur_alloc.count(b) || !open_ver[b]) {
      cur_alloc[b] = next_alloc++;
      open_ver[b] = true;
      S.has_readers.push_back(0);
      S.last_reader.emplace_back((size_t)-1, (size_t)-1);
    }
  };
  auto on_read = [&](int64_t b) {
    auto it = cur_alloc.find(b);
    if (it == cur_alloc.end()) return;
    S.has_readers[it->second] = 1;
    S.last_reader[it->second] = {item, pos};
    open_ver[b] = false;
  };
  auto in = [](const std::vector<int64_t>& v, int64_t b) {
    return std::find(v.begin(), v.end(), b) != v.end();
  };
  togsim::TraceSink sink = [&](const TraceRec& t) {
    pos = rec++;
    if (t.kind == TraceRec::DMA) {
      if (t.dir == 1) { for (int64_t b : t.read_bufs) on_read(b); }            // store drains the spad
      else            { for (int64_t b : t.write_bufs) on_open(b); }           // load fills it
    } else if (t.kind == TraceRec::COMPUTE) {
      for (int64_t b : t.read_bufs)  if (!in(t.write_bufs, b)) on_read(b);
      for (int64_t b : t.write_bufs) if (!in(t.read_bufs, b) && S.buf_bytes.count(b)) on_open(b);
    }
  };
  for (item = 0; item < S.prod.num_items(); item++) { rec = 0; S.prod.run_item(item, sink); }
}

// Materialize exactly one dispatch tile (work-item `S.next`) and return its
// subgraph; nullptr once the producer is exhausted.
std::shared_ptr<TileSubGraph> build_one_tile(BuildState& S) {
  using togsim::TraceRec;
  if (S.next >= S.prod.num_items()) return nullptr;
  auto& sg = S.sg; auto& tile = S.tile;
  auto& writers = S.writers; auto& seeds = S.seeds;
  auto& current_dma = S.current_dma; auto& bar_for_load = S.bar_for_load;
  auto& next_tag = S.next_tag; auto& cur_tile_group = S.cur_tile_group;
  auto& cur_tile_bufs = S.cur_tile_bufs; auto& cur_tile_footprint = S.cur_tile_footprint;
  auto& next_alloc = S.next_alloc; auto& cur_alloc = S.cur_alloc;
  auto& open_ver = S.open_ver;
  auto& buf_bytes = S.buf_bytes;
  auto rec_bytes = [](const TraceRec& t) {
    size_t numel = 1;
    for (auto d : t.dims) numel *= (size_t)d;
    return numel * (t.elem_bits / 8);
  };

  auto flush = [&]() {
    if (sg && tile) {
      tile->set_spad_footprint(cur_tile_footprint);   // distinct-buffer resident set (1- vs 2-dispatch)
      sg->add_tile(tile);
      tile->set_owner(sg);
      S.sg_out = sg;   // hand this tile to the consumer
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

  // --- SRAM-capacity tracking (buffer-version allocations, sec 10.4) ---
  // A coarse tile = one version of its buffer; the fine DMAs that fill it share
  // one allocation, freed once all the version's consumers have issued (refcount
  // -> 0). NOT reset in flush(): the spad is one physical per-core resource, so a
  // buffer reused by the next reduction iter / work-item is a NEW version that
  // must wait for the old one to free (WAR / double-buffer). Both DMA-loaded
  // buffers AND compute outputs (the accumulator, vector epilogue results) are
  // tracked; the virtual SA-weights are not (weight slots model them). (v1:
  // single-core; multi-core would key cur_alloc/vers by (core, buf).)
  // Add each buffer once to the current tile's footprint (reloads in a K-loop reuse the same id).
  auto note_bufs = [&](const std::vector<int64_t>& bufs) {
    for (int64_t b : bufs)
      if (cur_tile_bufs.insert(b).second) {
        auto it = buf_bytes.find(b);
        if (it != buf_bytes.end()) cur_tile_footprint += it->second;
      }
  };
  // a version nothing ever reads is never freed -> leave it untracked
  auto tracked = [&](int64_t a) { return S.has_readers[a] ? a : (int64_t)-1; };
  auto sram_on_load = [&](int64_t b, const std::shared_ptr<Instruction>& ld) {
    if (!cur_alloc.count(b) || !open_ver[b]) {   // a read closed it -> new version
      cur_alloc[b] = next_alloc++;
      open_ver[b] = true;
    }
    ld->set_sram_alloc(tracked(cur_alloc[b]));
  };
  // A compute that freshly produces b opens a version like a load, carrying b's
  // footprint; its last reader frees it -- identical lifecycle to a load.
  auto sram_on_write = [&](int64_t b, const std::shared_ptr<Instruction>& w) {
    auto bb = buf_bytes.find(b);
    if (bb == buf_bytes.end()) return;           // size unknown (never DMA'd) -> untracked
    if (!cur_alloc.count(b) || !open_ver[b]) {   // a consuming read closed it -> new version
      cur_alloc[b] = next_alloc++;
      open_ver[b] = true;
      w->set_sram_alloc(tracked(cur_alloc[b]));
      w->set_sram_footprint(bb->second);
    }
    // already-open version (further producing writes): same physical bytes, no re-add.
  };
  auto sram_on_read = [&](int64_t b, const std::shared_ptr<Instruction>& rd) {
    auto it = cur_alloc.find(b);
    if (it == cur_alloc.end()) return;           // not a load buffer -> untracked
    // The version's LAST reader frees it. sram_schedule() already found where
    // that reader sits, so tag it now rather than retaining every reader.
    if (S.last_reader[it->second] == std::make_pair(S.item, S.rec))
      rd->add_sram_release(it->second);
    open_ver[b] = false;                          // next write starts a new version
  };

  auto feed = [&](const TraceRec& t) {            // build, one record at a time
    struct RecTick { size_t& r; ~RecTick() { ++r; } } tick{S.rec};
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

  S.item = S.next; S.rec = 0;
  S.prod.run_item(S.next++, feed);
  auto out = std::move(S.sg_out);
  S.sg_out.reset();
  return out;
}

}  // namespace

std::unique_ptr<TileGraph> trace_to_tilegraph(
    const char* so_path, const int64_t* shape_args, int32_t n_shape,
    const uint64_t* tensor_base, int32_t n_tensors,
    const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
    const int32_t* partition_cores, int32_t n_partition_cores,
    const std::string& name) {
  using togsim::TraceRec;
  auto S = std::make_shared<BuildState>();

  // Indexing pass, which doubles as the spad-footprint pre-pass: one full
  // producer run that records each togsim_dispatch (tile fn, induction vars,
  // core) so the work-item can be re-invoked on its own later, while streaming
  // every record -- including any emitted outside a dispatch, which belong to no
  // work-item and never enter the graph -- into the footprint sink. It builds no
  // Instruction.
  auto rec_bytes = [](const TraceRec& t) {        // single source of the tile footprint
    size_t numel = 1;
    for (auto d : t.dims) numel *= (size_t)d;
    return numel * (t.elem_bits / 8);
  };
  togsim::TraceSink footprint = [&](const TraceRec& t) {
    if (t.kind != TraceRec::DMA) return;
    const auto& bs = (t.dir == 1) ? t.read_bufs : t.write_bufs;  // store reads spad, load writes spad
    for (int64_t b : bs) S->buf_bytes[b] = rec_bytes(t);
  };
  if (!S->prod.open(so_path, shape_args, n_shape, tensor_base, n_tensors,
                    cyc, ovl, n_tiles, partition_cores, n_partition_cores, &footprint))
    return nullptr;

  sram_schedule(*S);   // buffer-version lifetimes, materializing no Instruction

  auto tg = std::make_unique<TileGraph>(name, name);
  // Empty cache plan (no L2/CMEM persistence) -- the tile source propagates it
  // to each subgraph, and DMA::is_cacheable dereferences it, so it must be a
  // valid (if empty) IntervalTree rather than null.
  tg->init_cache_plan({});
  tg->set_tile_source([S]() { return build_one_tile(*S); });
  return tg;
}
