// togsim_trace_bridge.cc -- see togsim_trace_bridge.h
#include "togsim_trace_bridge.h"

#include <map>
#include <utility>
#include <vector>

#include "Tile.h"
#include "Instruction.h"

namespace {

// `uniq` is a per-DMA-record unique tag-key id minted by the caller. The Core
// tag table keys completion on [addr_id, ..., sum(tag_idx*stride)]; using `uniq`
// as addr_id makes every reduction iteration of one static dma get a DISTINCT
// key -- so multi-tile-K (and conv, whose reduction is the kh*kw*C nest) do not
// collide, with no coordinate enumeration. The matching memory_barrier reuses
// the same `uniq` (current-load map per (tag_id, tag_slot), see
// trace_to_tilegraph), so the table still pairs them. This works because the
// recorded stream is already per-iteration (the producer ran the loops) --
// unlike a compile-time event_id. `tag_idx` (the subtile slot) is retained for
// the SRAM double-buffer model.
//
// FIXME(semantics): the per-iteration tag is still reconstructed HERE from the
// record order. The producer IR now DOES carry a per-iteration tag -- dma_fine_-
// grained emits a fresh tag memref.alloc just before each coarse load (rewiring
// its dma_wait), so successive reduction iterations allocate distinct tags -- but
// build_skeleton collapses that to one static tag_id (it DCEs the alloc and keys
// togsim.dma by the alloc's static identity), so this bridge still needs `uniq`
// to tell iterations apart at runtime. The faithful finish is to thread the
// per-iteration alloc identity through build_skeleton as an SSA tag handle on the
// togsim.dma / togsim.memory_barrier (then `uniq` here is unnecessary).
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
// raw add_child which the async dma releases at issue-complete.
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

std::unique_ptr<TileGraph> trace_to_tilegraph(const togsim::RunResult& run,
                                              const std::string& name) {
  using togsim::TraceRec;
  auto tg = std::make_unique<TileGraph>(name, name);
  // Empty cache plan (no L2/CMEM persistence) -- append_subgraph propagates it
  // to each subgraph, and DMA::is_cacheable dereferences it, so it must be a
  // valid (if empty) IntervalTree rather than null.
  tg->init_cache_plan({});

  std::shared_ptr<TileSubGraph> sg;
  std::shared_ptr<Tile> tile;
  // Explicit dependency DAG (sec 10): a reader depends on the last writer of each
  // SRAM buffer it reads. Scoped per work-item (reset at each dispatch) -- buffers
  // are work-item-local, so distinct work-items are independent (-> parallel).
  std::map<int64_t, std::shared_ptr<Instruction>> last_writer;  // buffer id -> producer
  // An async dma is paired with its explicit memory_barrier(s) by the runtime tag
  // (tag_id, tag_slot). It is 1 load : N barriers (the load happens once per
  // reduction iteration; each consumer in that iteration is preceded by a wait on
  // the same tag), so we track the CURRENT (most recent) load per (tag_id,
  // tag_slot) -- like last_writer for a buffer -- not a FIFO. Each load gets a
  // fresh `uniq` Core key, so successive reduction iterations (multi-tile-K, conv)
  // never collide in the tag table; the iteration's barriers reuse that load's
  // uniq. Correct because the load nest and its consumer nest run in order within
  // the reduction body (no cross-iteration prefetch). Scoped per work-item.
  std::map<std::pair<int32_t, uint64_t>,
           std::pair<int64_t, std::shared_ptr<Instruction>>> current_dma;
  // Dedup identical dma_waits: the barrier already built for the CURRENT load of a
  // (tag_id, tag_slot). A later memory_barrier on the SAME load instance reuses it
  // (its consumers gate on the existing bar) instead of re-emitting -- a conv reads one
  // loaded subtile from many matmuls, so the fine-grained per-consumer waits collapse to
  // one per load. A new load (next reduction iter) bumps uniq, so a genuine new wait
  // still gets its own bar; the first wait stays at its consumer, so overlap is kept.
  std::map<std::pair<int32_t, uint64_t>,
           std::pair<int64_t, std::shared_ptr<Instruction>>> bar_for_load;
  int64_t next_tag = 0;   // mints a unique Core tag key per dma record
  int cur_tile_group = -1;   // work-item index, bumped per TILE_BEGIN (trace grouping)
  // Async compute (matmul/preload): issued and pipelined on the systolic array;
  // they do not block each other. A store then needs the drained result, so it
  // FLUSHes -- waits all outstanding async compute before running (like a fence
  // after async ops). No per-op completion events; one barrier before the store.
  std::vector<std::shared_ptr<Instruction>> outstanding_async;
  std::shared_ptr<Instruction> pending_bar;   // last COMPUTE_BAR fence, awaited by the next store
  auto is_async_compute = [](int ct) { return ct == 1 || ct == 2; };  // matmul / preload

  auto flush = [&]() {
    if (sg && tile) {
      sg->add_tile(tile);
      tile->set_owner(sg);
      tg->append_subgraph(sg);
    }
    sg.reset();
    tile.reset();
    last_writer.clear();
    current_dma.clear();
    bar_for_load.clear();
    next_tag = 0;
    outstanding_async.clear();
    pending_bar.reset();
  };

  // Build edges from the recorded read/write buffer sets: reader <- last writer of
  // each buffer it reads (the virtual SA_WEIGHTS buffer carries preload->matmul;
  // the Y_spad accumulator carries the reduction chain; the spads carry load->
  // compute). No in-order chain, no tag matching, no op heuristics.
  // sec 10.7 occupancy/latency split. An edge from a systolic-array producer
  // (preload=2 or matmul=1) to a matmul (1) is an OCCUPANCY dependency: the
  // successor overlaps the producer on the SA pipeline, so use add_pipeline_child
  // (released when the producer ISSUES). Every other edge is a LATENCY
  // dependency (the consumer needs the producer's result): load->compute,
  // init->matmul, matmul->store -> add_child (released at the producer's finish).
  const int MATMUL_CT = 1, PRELOAD_CT = 2;
  auto link = [&](std::shared_ptr<Instruction> inst,
                  const std::vector<int64_t>& reads,
                  const std::vector<int64_t>& writes) {
    for (int64_t b : reads) {
      // A matmul reading its own accumulator (a buffer it also WRITES) imposes NO
      // producer order: Y += X@W is commutative. Chaining matmuls through the
      // accumulator (M_k <- M_{k-1}) needlessly serializes them and DEADLOCKS the SA
      // weight-slot pipeline -- a later iteration's preload can grab the last weight
      // slot while the in-order head matmul is starved of one, and that head can never
      // run to release a slot. The store still waits every matmul via the COMPUTE_BAR
      // fence, so dropping this edge is safe (TOGSim is timing-only; values come from
      // the recorded trace).
      bool is_accum = false;
      for (int64_t w : writes) if (w == b) { is_accum = true; break; }
      if (inst->get_compute_type() == MATMUL_CT && is_accum) continue;
      auto it = last_writer.find(b);
      if (it == last_writer.end()) continue;
      int pct = it->second->get_compute_type();
      if (inst->get_compute_type() == MATMUL_CT && (pct == MATMUL_CT || pct == PRELOAD_CT))
        it->second->add_pipeline_child(inst);  // SA pipeline -> occupancy (overlap)
      else
        it->second->add_child(inst);           // data/result -> latency (full wait)
    }
    for (int64_t b : writes) last_writer[b] = inst;
    tile->append_instuction(inst);
  };

  // --- SRAM-capacity tracking (buffer-version allocations, sec 10.x) ---
  // A coarse tile = one version of its buffer; the fine DMAs that fill it share
  // one allocation, freed once all the version's consumers have issued (refcount
  // -> 0). NOT reset in flush(): the spad is one physical per-core resource, so a
  // buffer reused by the next reduction iter / work-item is a NEW version that
  // must wait for the old one to free (WAR / double-buffer). Tracked buffers are
  // the DMA-loaded ones; the accumulator / virtual SA-weights are never written
  // by a load, so cur_alloc has no entry and they are skipped. (v1: single-core;
  // multi-core would key cur_alloc/vers by (core, buf).)
  int64_t next_alloc = 0;
  std::map<int64_t, int64_t> cur_alloc;   // buf -> current version id
  std::map<int64_t, bool> open_ver;       // buf -> version still accepting loads
  struct Ver { std::vector<std::shared_ptr<Instruction>> loads, readers; };
  std::map<int64_t, Ver> vers;
  auto sram_on_load = [&](int64_t b, const std::shared_ptr<Instruction>& ld) {
    if (!cur_alloc.count(b) || !open_ver[b]) {   // a read closed it -> new version
      cur_alloc[b] = next_alloc++;
      open_ver[b] = true;
      vers[cur_alloc[b]] = {};
    }
    ld->set_sram_alloc(cur_alloc[b]);
    vers[cur_alloc[b]].loads.push_back(ld);
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

  for (const auto& t : run.trace) {
    if (t.kind == TraceRec::TILE_BEGIN) {
      // togsim_dispatch opened a work-item -> new subgraph (bound to its core) +
      // tile. The scope runs until the matching TILE_END (the dispatch wrapper
      // brackets the tile fn call), not until the next begin.
      flush();
      sg = std::make_shared<TileSubGraph>();
      sg->set_core_id(t.core);
      tile = std::make_shared<Tile>(Tile::Status::INITIALIZED);
      cur_tile_group++;
      continue;
    }
    if (t.kind == TraceRec::TILE_END) {
      flush();   // close the work-item explicitly (scope = the tile fn call)
      continue;
    }
    if (!tile) continue;  // defensive: ops before the first TILE_BEGIN

    if (t.kind == TraceRec::DMA) {
      int64_t uniq = next_tag++;                         // fresh Core tag key per dma record
      auto inst = make_dma(t, uniq);
      inst->set_tile_group(cur_tile_group);
      size_t numel = 1;                                  // SRAM footprint (ready-tile ordering)
      for (auto d : t.dims) numel *= (size_t)d;
      tile->inc_required_sram_size(numel * (t.elem_bits / 8));
      if (t.dir == 1) {                                  // STORE
        if (pending_bar) {
          // after a compute fence: wait it (drains the async matmuls) -- covers
          // the accumulator read, so no per-buffer read edge.
          pending_bar->add_child(inst);
          pending_bar.reset();
          for (int64_t b : t.write_bufs) last_writer[b] = inst;
          tile->append_instuction(inst);
        } else {
          link(inst, t.read_bufs, t.write_bufs);
        }
        for (int64_t b : t.read_bufs) sram_on_read(b, inst);  // store frees what it drains
      } else {                                           // LOAD
        tile->append_instuction(inst);
        // async load: record it as the CURRENT load for this (tag_id, tag_slot)
        // with its fresh uniq; the barriers in this reduction iteration reuse that
        // uniq (1 load : N barriers). A new iteration's load overwrites it with a
        // new uniq -> distinct tag key, no collision. last_writer = the dma for now;
        // the barrier overwrites it so consumers gate on data arrival. A sync load
        // has no barrier and blocks to arrival itself.
        if (t.is_async) current_dma[{t.tag_id, t.tag_slot}] = {uniq, inst};
        for (int64_t b : t.write_bufs) last_writer[b] = inst;
        for (int64_t b : t.write_bufs) sram_on_load(b, inst);   // occupy spad
      }
    } else if (t.kind == TraceRec::MEMORY_BAR) {
      // the explicit async-DMA sync (the original dma_wait). Pair with the CURRENT
      // load for this (tag_id, tag_slot), reusing its uniq Core key so the dma and
      // bar pair in the tag table; the dma releases the bar at issue-complete
      // (add_child), then the bar parks on the tag until data-ready (resp-complete,
      // set_tag_finish). Consumers of the loaded buffer then gate on the bar.
      auto it = current_dma.find({t.tag_id, t.tag_slot});
      int64_t uniq = next_tag++;                         // fallback if unpaired
      std::shared_ptr<Instruction> dma_inst;
      if (it != current_dma.end()) { uniq = it->second.first; dma_inst = it->second.second; }
      // Identical wait (same slot, same load instance) already has a barrier -> reuse it
      // so the buffer's consumers gate on it, instead of emitting a redundant barrier.
      auto bf = bar_for_load.find({t.tag_id, t.tag_slot});
      if (bf != bar_for_load.end() && bf->second.first == uniq) {
        for (int64_t b : t.write_bufs) last_writer[b] = bf->second.second;
        continue;
      }
      auto bar = make_mem_bar(t, uniq);
      bar->set_tile_group(cur_tile_group);
      if (dma_inst) dma_inst->add_child(bar);
      tile->append_instuction(bar);
      for (int64_t b : t.write_bufs) last_writer[b] = bar;
      bar_for_load[{t.tag_id, t.tag_slot}] = {uniq, bar};
    } else if (t.kind == TraceRec::COMPUTE) {
      auto inst = make_compute(t);
      inst->set_tile_group(cur_tile_group);
      link(inst, t.read_bufs, t.write_bufs);
      for (int64_t b : t.read_bufs) sram_on_read(b, inst);     // frees the tiles it consumes
      if (is_async_compute(t.compute_type)) outstanding_async.push_back(inst);
    } else if (t.kind == TraceRec::COMPUTE_BAR) {
      // explicit compute fence: ready once all outstanding async compute have
      // ISSUED (pipeline-child release); the Core then waits the SA pipelines to
      // drain before it finishes (-> the store it gates).
      auto bar = std::make_shared<Instruction>(Opcode::COMPUTE_BAR);
      bar->set_tile_group(cur_tile_group);
      for (auto& a : outstanding_async) a->add_pipeline_child(bar);
      outstanding_async.clear();
      tile->append_instuction(bar);
      pending_bar = bar;
    }
  }
  flush();
  sram_finalize();   // readers per version are now final -> set each version's refcount
  return tg;
}
