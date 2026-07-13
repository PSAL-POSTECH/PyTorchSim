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

// Core's compute-unit enum, as the producer encodes it in TraceRec::compute_type.
constexpr int MATMUL_CT = 1, PRELOAD_CT = 2;

// The current producers of one SRAM buffer. Normally a write REPLACEs them, but
// the K matmuls of Y += X@W commute: each only waits on whoever INITIALIZED the
// accumulator, so it JOINs the set and reads initializer() in O(1), not O(K).
class BufferWriters {
 public:
  void replace(const std::shared_ptr<Instruction>& w) {   // normal write: sole producer
    _all.assign(1, w);
    _initializer = (w->get_compute_type() != MATMUL_CT) ? w : nullptr;
  }
  void accumulate(const std::shared_ptr<Instruction>& mm) { _all.push_back(mm); }

  const std::vector<std::shared_ptr<Instruction>>& all() const { return _all; }
  const std::shared_ptr<Instruction>& initializer() const { return _initializer; }

 private:
  std::vector<std::shared_ptr<Instruction>> _all;
  std::shared_ptr<Instruction> _initializer;   // the non-MATMUL member of _all, if any
};

// All builder state for one kernel, driven one dispatch tile at a time. Owned by
// the TileGraph's tile source, so it survives between materializations. link() is
// the dependency DAG, sram_effects() the buffer-version rule, feed() the dispatch.
struct BuildState {
  togsim::LazyProducer prod;
  size_t next = 0;               // next work-item to materialize
  std::map<int64_t, size_t> buf_bytes;
  std::shared_ptr<TileSubGraph> sg_out;   // the tile just built

  std::shared_ptr<TileSubGraph> sg;
  std::shared_ptr<Tile> tile;
  std::map<int64_t, BufferWriters> writers;
  std::map<std::pair<int32_t, uint64_t>,
           std::pair<int64_t, std::shared_ptr<Instruction>>> current_dma;
  std::map<std::pair<int32_t, uint64_t>,
           std::pair<int64_t, std::shared_ptr<Instruction>>> bar_for_load;
  int64_t next_tag = 0;
  int cur_tile_group = -1;
  std::set<int64_t> cur_tile_bufs;
  size_t cur_tile_footprint = 0;

  // SRAM buffer versions, NOT scoped to a work-item: the spad is one physical
  // resource, so a buffer reused by the next tile is a new version that waits for
  // the old one to free. sram_schedule() precomputes the schedule; see below.
  int64_t next_alloc = 0;
  std::map<int64_t, int64_t> cur_alloc;   // buf -> current version id
  std::map<int64_t, bool> open_ver;       // buf -> version still accepting writes
  std::vector<char> has_readers;                        // version -> ever read?
  std::vector<std::pair<size_t, size_t>> last_reader;   // version -> (work-item, record)
  size_t item = 0, rec = 0;               // position of the record being fed

  static size_t rec_bytes(const togsim::TraceRec& t) {   // single source of the tile footprint
    size_t numel = 1;
    for (auto d : t.dims) numel *= (size_t)d;
    return numel * (t.elem_bits / 8);
  }

  // ---- index + footprint --------------------------------------------------
  // Open the producer, then replay every work-item once to size each buffer:
  // buf_bytes must be known before sram_schedule() and before the build.
  bool index(const char* so_path, const int64_t* shape_args, int32_t n_shape,
             const uint64_t* tensor_base, int32_t n_tensors,
             const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
             const int32_t* partition_cores, int32_t n_partition_cores) {
    using togsim::TraceRec;
    if (!prod.open(so_path, shape_args, n_shape, tensor_base, n_tensors,
                   cyc, ovl, n_tiles, partition_cores, n_partition_cores))
      return false;
    for (size_t i = 0; i < prod.num_items(); i++)
      for (const TraceRec& t : prod.run_item(i)) {
        if (t.kind != TraceRec::DMA) continue;
        const auto& bs = (t.dir == 1) ? t.read_bufs : t.write_bufs;  // store reads spad, load writes spad
        for (int64_t b : bs) buf_bytes[b] = rec_bytes(t);
      }
    return true;
  }

  // ---- the SRAM buffer-version rule (THE single source of truth) -----------
  // What `t` does to the spad: `reads` consume (and close) a buffer's current
  // version, `opens` may start a new one. sram_schedule() and feed() must agree on
  // this or version ids drift, so both call it. Apply `reads` before `opens`.
  void sram_effects(const togsim::TraceRec& t,
                    std::vector<int64_t>& reads,
                    std::vector<int64_t>& opens) const {
    using togsim::TraceRec;
    reads.clear();
    opens.clear();
    auto has = [](const std::vector<int64_t>& v, int64_t b) {
      return std::find(v.begin(), v.end(), b) != v.end();
    };
    if (t.kind == TraceRec::DMA) {
      if (t.dir == 1) reads = t.read_bufs;    // store drains the spad
      else            opens = t.write_bufs;   // load fills it
    } else if (t.kind == TraceRec::COMPUTE) {
      // An in-place buffer (read AND written: the accumulator, an in-place
      // vector op) is version-transparent -- it neither closes nor opens one.
      for (int64_t b : t.read_bufs)
        if (!has(t.write_bufs, b)) reads.push_back(b);          // consuming reads
      for (int64_t b : t.write_bufs)
        if (!has(t.read_bufs, b) && buf_bytes.count(b))         // fresh outputs; a
          opens.push_back(b);                                   // never-DMA'd buf has
    }                                                           // no size -> untracked
  }

  // ---- version lifetimes, precomputed (allocates no Instruction) -----------
  // Per buffer version, record whether anything reads it and where its LAST reader
  // sits, so feed() can tag that reader as it goes rather than retain every reader.
  void sram_schedule() {
    using togsim::TraceRec;
    int64_t alloc = 0;
    std::map<int64_t, int64_t> cur;    // buf -> current version id
    std::map<int64_t, bool> open;      // buf -> version still accepting writes
    std::vector<int64_t> reads, opens;

    for (size_t wi = 0; wi < prod.num_items(); wi++) {
      size_t pos = 0;
      for (const TraceRec& t : prod.run_item(wi)) {
        sram_effects(t, reads, opens);
        for (int64_t b : reads) {
          auto f = cur.find(b);
          if (f == cur.end()) continue;              // untracked buffer
          has_readers[f->second] = 1;
          last_reader[f->second] = {wi, pos};        // keep the LAST one
          open[b] = false;                           // next write starts a new version
        }
        for (int64_t b : opens)
          if (!cur.count(b) || !open[b]) {           // a read closed it -> new version
            cur[b] = alloc++;
            open[b] = true;
            has_readers.push_back(0);
            last_reader.emplace_back((size_t)-1, (size_t)-1);
          }
        pos++;
      }
    }
  }

  // ---- per-tile close -----------------------------------------------------
  void flush() {
    if (sg && tile) {
      tile->set_spad_footprint(cur_tile_footprint);   // distinct-buffer resident set (1- vs 2-dispatch)
      sg->add_tile(tile);
      tile->set_owner(sg);
      sg_out = sg;   // hand this tile to the consumer
    }
    sg.reset();
    tile.reset();
    writers.clear();
    current_dma.clear();
    bar_for_load.clear();
    cur_tile_bufs.clear();
    cur_tile_footprint = 0;
    next_tag = 0;
  }

  // ---- dependency DAG (sec 10.3) ------------------------------------------
  // READ b: depend on all writers(b), ISSUE when both are SA ops else DONE.
  // WRITE b: replace writers(b), except a commutative matmul accumulator, which
  // joins them and waits only the initializer. WAR: resource models.
  bool is_mm_accum(const std::shared_ptr<Instruction>& inst, int64_t b,
                   const std::vector<int64_t>& writes) {
    if (inst->get_compute_type() != MATMUL_CT) return false;
    for (int64_t w : writes) if (w == b) return true;
    return false;
  }
  void link(std::shared_ptr<Instruction> inst,
            const std::vector<int64_t>& reads,
            const std::vector<int64_t>& writes) {
    for (int64_t b : reads) {
      if (is_mm_accum(inst, b, writes)) continue;   // accumulator read -> handled in WRITE (UNION)
      auto it = writers.find(b);
      if (it != writers.end())
        for (auto& w : it->second.all()) {
          int pct = w->get_compute_type();
          // both SA ops -> occupancy (overlap on the SA pipeline); else latency.
          DepEvent on = (inst->get_compute_type() == MATMUL_CT &&
                         (pct == MATMUL_CT || pct == PRELOAD_CT))
                            ? DepEvent::ISSUE : DepEvent::DONE;
          w->add_dep(inst, on);
        }
    }
    for (int64_t b : writes) {
      auto& w = writers[b];
      if (is_mm_accum(inst, b, writes)) {   // JOIN: commutative, waits only the init
        if (const auto& init = w.initializer()) init->add_dep(inst, DepEvent::DONE);
        w.accumulate(inst);
      } else {                              // REPLACE: a normal output resets the set
        w.replace(inst);
      }
    }
    tile->append_instuction(inst);
  }

  // ---- SRAM-capacity tracking (buffer-version allocations, sec 10.4) -------
  // A coarse tile = one version of its buffer; the fine DMAs filling it share one
  // allocation, freed once every consumer has issued. Tracks DMA-loaded buffers and
  // compute outputs, but not the virtual SA-weights (weight slots model those).
  void note_bufs(const std::vector<int64_t>& bufs) {
    for (int64_t b : bufs)   // once per buffer: a K-loop reload reuses the same id
      if (cur_tile_bufs.insert(b).second) {
        auto it = buf_bytes.find(b);
        if (it != buf_bytes.end()) cur_tile_footprint += it->second;
      }
  }
  // a version nothing ever reads is never freed -> leave it untracked
  int64_t tracked(int64_t a) { return has_readers[a] ? a : (int64_t)-1; }

  // Run record `t`'s spad effects against the live version state and tag `inst`
  // with what it allocates and frees. Same rule as sram_schedule() (sram_effects),
  // so the version ids agree; this pass just also touches the Instruction.
  void sram_apply(const togsim::TraceRec& t, const std::shared_ptr<Instruction>& inst) {
    using togsim::TraceRec;
    std::vector<int64_t> reads, opens;
    sram_effects(t, reads, opens);

    for (int64_t b : reads) {                      // consume the current version
      auto it = cur_alloc.find(b);
      if (it == cur_alloc.end()) continue;         // untracked buffer
      // Its LAST reader frees it. sram_schedule() already found where that reader
      // sits, so tag it here rather than retaining every reader to end-of-stream.
      if (last_reader[it->second] == std::make_pair(item, rec))
        inst->add_sram_release(it->second);
      open_ver[b] = false;                         // next write starts a new version
    }
    for (int64_t b : opens) {
      const bool fresh = !cur_alloc.count(b) || !open_ver[b];   // a read closed it
      if (fresh) {
        cur_alloc[b] = next_alloc++;
        open_ver[b] = true;
      }
      if (t.kind == TraceRec::DMA) {
        // Every fine DMA that fills the buffer carries the version id, so the
        // reloads of a reduction's K-loop share one allocation. Its footprint is
        // derived from the tile it moves, so nothing to set here.
        inst->set_sram_alloc(tracked(cur_alloc[b]));
      } else if (fresh) {
        // A compute that freshly produces b opens a version like a load and must
        // carry b's size explicitly (it was never DMA'd in). Further producing
        // writes into an already-open version are the same physical bytes.
        inst->set_sram_alloc(tracked(cur_alloc[b]));
        inst->set_sram_footprint(buf_bytes.at(b));
      }
    }
  }

  // ---- the record consumer (one record at a time) -------------------------
  void feed(const togsim::TraceRec& t) {
    using togsim::TraceRec;
    struct RecTick { size_t& r; ~RecTick() { ++r; } } tick{rec};
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
      } else {                                           // LOAD
        tile->append_instuction(inst);
        // async load: the CURRENT load for this (tag_id, tag_slot), with a fresh
        // uniq its barriers reuse. writers = the dma until its barrier overwrites it,
        // so consumers gate on arrival. A sync load blocks to arrival itself.
        if (t.is_async) current_dma[{t.tag_id, t.tag_slot}] = {uniq, inst};
        // No hard WAR edge: load-buffer reuse is modeled by the SRAM version /
        // capacity machinery (sram_apply), which caps how many versions coexist. A
        // latency WAR edge would force single-buffering and kill the spad overlap.
        for (int64_t b : t.write_bufs) writers[b].replace(inst);
      }
      sram_apply(t, inst);   // a store frees what it drains; a load occupies the spad
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
        for (int64_t b : t.write_bufs) writers[b].replace(bf->second.second);
        return;
      }
      auto bar = make_mem_bar(t, uniq);
      bar->set_tile_group(cur_tile_group);
      if (dma_inst) dma_inst->add_dep(bar, DepEvent::DONE);
      tile->append_instuction(bar);
      // the bar is the load's DONE-handle: REPLACE writers(b) with it (no WAR -- the
      // load already WAR'd the prior readers when it wrote).
      for (int64_t b : t.write_bufs) writers[b].replace(bar);
      bar_for_load[{t.tag_id, t.tag_slot}] = {uniq, bar};
    } else if (t.kind == TraceRec::COMPUTE) {
      auto inst = make_compute(t);
      inst->set_tile_group(cur_tile_group);
      link(inst, t.read_bufs, t.write_bufs);
      note_bufs(t.read_bufs); note_bufs(t.write_bufs);   // distinct-buffer footprint for 1- vs 2-dispatch
      sram_apply(t, inst);   // consuming reads free their version; fresh outputs open one
    }
  }

  // ---- materialize exactly one dispatch tile ------------------------------
  // Returns the work-item `next`'s subgraph; nullptr once the producer is
  // exhausted. Called from the TileGraph's tile source, on demand.
  std::shared_ptr<TileSubGraph> build_one_tile() {
    if (next >= prod.num_items()) return nullptr;
    item = next; rec = 0;
    for (const togsim::TraceRec& t : prod.run_item(next++)) feed(t);
    auto out = std::move(sg_out);
    sg_out.reset();
    return out;
  }
};

}  // namespace

std::unique_ptr<TileGraph> trace_to_tilegraph(
    const char* so_path, const int64_t* shape_args, int32_t n_shape,
    const uint64_t* tensor_base, int32_t n_tensors,
    const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
    const int32_t* partition_cores, int32_t n_partition_cores,
    const std::string& name) {
  using togsim::TraceRec;
  auto S = std::make_shared<BuildState>();

  // Index the dispatches (records each work-item's fn/iv/core) and collect each
  // buffer's spad size. Builds no Instruction.
  if (!S->index(so_path, shape_args, n_shape, tensor_base, n_tensors,
                cyc, ovl, n_tiles, partition_cores, n_partition_cores))
    return nullptr;

  S->sram_schedule();   // buffer-version lifetimes, materializing no Instruction

  auto tg = std::make_unique<TileGraph>(name, name);
  // Empty cache plan (no L2/CMEM persistence) -- the tile source propagates it
  // to each subgraph, and DMA::is_cacheable dereferences it, so it must be a
  // valid (if empty) IntervalTree rather than null.
  tg->init_cache_plan({});
  tg->set_tile_source([S]() { return S->build_one_tile(); });
  return tg;
}
