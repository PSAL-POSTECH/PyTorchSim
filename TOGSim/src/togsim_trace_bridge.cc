// togsim_trace_bridge.cc -- see togsim_trace_bridge.h
#include "togsim_trace_bridge.h"

#include <map>
#include <utility>
#include <vector>

#include "Tile.h"
#include "Instruction.h"

namespace {

std::shared_ptr<Instruction> make_dma(const togsim::TraceRec& t) {
  Opcode op = (t.dir == 1) ? Opcode::MOVOUT : Opcode::MOVIN;
  std::vector<size_t> tile_size(t.dims.begin(), t.dims.end());
  std::vector<int> tile_stride(t.strides.begin(), t.strides.end());
  // tag_idx_list / tag_stride_list must match in size; one slot key per dma.
  std::vector<int64_t> tag_idx{(int64_t)t.tag_slot};
  std::vector<int64_t> tag_stride{1};
  auto inst = std::make_shared<Instruction>(
      op, /*compute_cycle=*/0, /*num_parents=*/0, /*dram_addr=*/t.addr,
      tile_size, tile_stride, (size_t)t.elem_bits, tag_idx, tag_stride,
      /*accum_tag_idx_list=*/std::vector<int64_t>{});
  inst->set_is_async(t.is_async != 0);
  // The tag key is [addr_id, ..., sum(tag_idx*tag_stride)]. addr_id is the tag
  // memref identity (tag_id): an async dma and its memory_barrier share a tag
  // memref, so the same (tag_id, tag_slot) keys both and the Core's tag table
  // pairs them. (Distinct tag memrefs -> distinct tag_id, so no false collision.)
  inst->set_addr_name("tag" + std::to_string(t.tag_id), t.tag_id);
  inst->prepare_tag_key();
  return inst;
}

// A MEMORY_BAR carrying the SAME tag key as the async dma it gates -- the Core's
// tag table signals it at the dma's DATA-ready (resp-complete), unlike a raw
// add_child which the async dma releases at issue-complete. Tag inputs match
// make_dma (tag_idx={tag_slot}, stride={1}, addr_id=tag_id) so the keys collide.
std::shared_ptr<Instruction> make_mem_bar(const togsim::TraceRec& t) {
  auto bar = std::make_shared<Instruction>(
      Opcode::MEMORY_BAR, 0, 0, 0,
      std::vector<size_t>{}, std::vector<int>{}, 0,
      std::vector<int64_t>{(int64_t)t.tag_slot}, std::vector<int64_t>{1},
      std::vector<int64_t>{});
  bar->set_addr_name("tag" + std::to_string(t.tag_id), t.tag_id);
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
  // An async dma is paired with its explicit memory_barrier by the RUNTIME tag
  // (tag_id, tag_slot): the dma records itself here so the later barrier can find
  // it and depend on it. Scoped per work-item (the tag table is per subgraph).
  std::map<std::pair<int32_t, uint64_t>, std::shared_ptr<Instruction>> tag_to_dma;
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
    tag_to_dma.clear();
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
    if (t.kind == TraceRec::DISPATCH) {
      // new work-item -> new subgraph (bound to its core) + tile.
      flush();
      sg = std::make_shared<TileSubGraph>();
      sg->set_core_id(t.core);
      tile = std::make_shared<Tile>(Tile::Status::INITIALIZED);
      continue;
    }
    if (!tile) continue;  // defensive: ops before the first core_alloc

    if (t.kind == TraceRec::DMA) {
      auto inst = make_dma(t);
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
        // Record the dma under its runtime tag so the explicit memory_barrier
        // (the original dma_wait) can pair with it. last_writer = the dma for now;
        // for an async load the barrier overwrites it (consumers gate on data
        // arrival), for a sync load the dma itself blocks to data arrival.
        if (t.is_async) tag_to_dma[{t.tag_id, t.tag_slot}] = inst;
        for (int64_t b : t.write_bufs) last_writer[b] = inst;
      }
    } else if (t.kind == TraceRec::MEMORY_BAR) {
      // the explicit async-DMA sync (the original dma_wait). Pair with its dma by
      // the runtime tag; the dma releases the bar at issue-complete (add_child),
      // then the bar parks on the tag table until the data arrives (resp-complete,
      // set_tag_finish). Consumers of the loaded buffer then gate on the bar.
      auto bar = make_mem_bar(t);
      auto it = tag_to_dma.find({t.tag_id, t.tag_slot});
      if (it != tag_to_dma.end()) it->second->add_child(bar);
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
