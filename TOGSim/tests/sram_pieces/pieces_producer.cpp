// A 16 KB load buffer filled by four 4 KB pieces, reloaded once after a consuming read,
// and a 16 KB output drained by four 4 KB stores: each buffer counts 16 KB (one version),
// so a work-item's spad footprint is 32 KB. Two work-items share the spad.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64
#define PIECE_ROWS 16

static void pieces_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t piece[2] = {PIECE_ROWS, COLS}, st[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2}, both[2] = {1, 2};
  const uint64_t base = (uint64_t)iv[0] * ROWS * COLS;
  for (int k = 0; k < 2; k++) {
    for (int p = 0; p < ROWS / PIECE_ROWS; p++)
      togsim_dma(ctx, 0, 0, base + p * PIECE_ROWS * COLS, 2, piece, st, 32, 0, 0, nullptr, 0, in, 1, 0);
    if (k == 0) togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
    else        togsim_compute(ctx, 0, 0, 0, nullptr, both, 2, out, 1);
  }
  for (int p = 0; p < ROWS / PIECE_ROWS; p++)
    togsim_dma(ctx, 1, 1, base + p * PIECE_ROWS * COLS, 2, piece, st, 32, 0, 0, out, 1, nullptr, 0, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < 2; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, pieces_tile, iv, 1, i);
  }
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
