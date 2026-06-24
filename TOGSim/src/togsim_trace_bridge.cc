// togsim_trace_bridge.cc -- see togsim_trace_bridge.h
#include "togsim_trace_bridge.h"

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
  // 1 load : N barriers, so track the CURRENT load per (tag_id, tag_slot), not a
  // FIFO. Each load takes a fresh `uniq` Core key and its iteration's barriers reuse
  // it. Correct only because a load nest and its consumers run in order. Per work-item.
  std::map<std::pair<int32_t, uint64_t>,
           std::pair<int64_t, std::shared_ptr<Instruction>>> current_dma;
  int64_t next_tag = 0;   // mints a unique Core tag key per dma record
  // Async compute (matmul/preload) pipelines on the systolic array. A store needs the
  // drained result, so it FLUSHes -- one barrier before the store waits all outstanding
  // async compute, with no per-op completion events.
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
    next_tag = 0;
    outstanding_async.clear();
    pending_bar.reset();
  };

  // Edges from the recorded read/write buffer sets: a reader depends on the last writer
  // of each buffer it reads. An SA-producer -> matmul edge is an OCCUPANCY dependency
  // (released at ISSUE); every other edge is a LATENCY dependency (released at finish).
  const int MATMUL_CT = 1, PRELOAD_CT = 2;
  auto link = [&](std::shared_ptr<Instruction> inst,
                  const std::vector<int64_t>& reads,
                  const std::vector<int64_t>& writes) {
    for (int64_t b : reads) {
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

  for (const auto& t : run.trace) {
    if (t.kind == TraceRec::TILE_BEGIN) {
      // togsim_dispatch opened a work-item -> new subgraph (bound to its core) +
      // tile. The scope runs until the matching TILE_END (the dispatch wrapper
      // brackets the tile fn call), not until the next begin.
      flush();
      sg = std::make_shared<TileSubGraph>();
      sg->set_core_id(t.core);
      tile = std::make_shared<Tile>(Tile::Status::INITIALIZED);
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
      } else {                                           // LOAD
        tile->append_instuction(inst);
        // async load: the CURRENT load for this (tag_id, tag_slot), with a fresh uniq
        // its barriers reuse. last_writer = the dma until its barrier overwrites it, so
        // consumers gate on arrival. A sync load blocks to arrival itself.
        if (t.is_async) current_dma[{t.tag_id, t.tag_slot}] = {uniq, inst};
        for (int64_t b : t.write_bufs) last_writer[b] = inst;
      }
    } else if (t.kind == TraceRec::MEMORY_BAR) {
      // The explicit async-DMA sync. Pair with the CURRENT load for this (tag_id,
      // tag_slot), reusing its uniq: the dma releases the bar at issue, the bar parks on
      // the tag until resp-complete, and consumers of the buffer gate on the bar.
      auto it = current_dma.find({t.tag_id, t.tag_slot});
      int64_t uniq = next_tag++;                         // fallback if unpaired
      std::shared_ptr<Instruction> dma_inst;
      if (it != current_dma.end()) { uniq = it->second.first; dma_inst = it->second.second; }
      auto bar = make_mem_bar(t, uniq);
      if (dma_inst) dma_inst->add_child(bar);
      tile->append_instuction(bar);
      for (int64_t b : t.write_bufs) last_writer[b] = bar;
    } else if (t.kind == TraceRec::COMPUTE) {
      auto inst = make_compute(t);
      link(inst, t.read_bufs, t.write_bufs);
      if (is_async_compute(t.compute_type)) outstanding_async.push_back(inst);
    } else if (t.kind == TraceRec::COMPUTE_BAR) {
      // explicit compute fence: ready once all outstanding async compute have
      // ISSUED (pipeline-child release); the Core then waits the SA pipelines to
      // drain before it finishes (-> the store it gates).
      auto bar = std::make_shared<Instruction>(Opcode::COMPUTE_BAR);
      for (auto& a : outstanding_async) a->add_pipeline_child(bar);
      outstanding_async.clear();
      tile->append_instuction(bar);
      pending_bar = bar;
    }
  }
  flush();
  return tg;
}
