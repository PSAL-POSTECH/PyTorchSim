#pragma once
// A trace tensor's DRAM layout for TOGSim's timing only: its (rows, cols) view stored as
// (tile_rows, tile_cols) tiles, tiles row-major and row-major inside each. Spike and the
// functional run keep every tensor row-major; only the requests a DMA sends DRAM move.
#include <cstdint>

struct TensorTiling {
  uint64_t base, rows, cols, elem_bytes, tile_rows, tile_cols;

  // Where the element the row-major byte address `a` names sits in the tiled tensor.
  uint64_t map(uint64_t a) const {
    const uint64_t off = a - base, i = off / elem_bytes, r = i / cols, c = i % cols;
    const uint64_t tile = (r / tile_rows) * tiles_per_row() + c / tile_cols;
    return base + elem_bytes * (tile * tile_rows * tile_cols + (r % tile_rows) * tile_cols + c % tile_cols)
           + off % elem_bytes;
  }
  uint64_t tiles_per_row() const { return (cols + tile_cols - 1) / tile_cols; }
  // Bytes the tensor spans once padded to whole tiles.
  uint64_t footprint() const {
    return (rows + tile_rows - 1) / tile_rows * tiles_per_row() * tile_rows * tile_cols * elem_bytes;
  }
};
