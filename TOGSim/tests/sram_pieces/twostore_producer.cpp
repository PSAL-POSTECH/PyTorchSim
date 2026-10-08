// One 16 KB output computed once and stored to two DRAM tensors: the second store drains the same
// 16 KB, so a work-item's spad footprint is 32 KB (input + output), not 48 KB.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64

static void twostore_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t d[2] = {ROWS, COLS}, st[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  const uint64_t base = (uint64_t)iv[0] * ROWS * COLS;
  togsim_dma(ctx, 0, 0, base, 2, d, st, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, 1, 1, base, 2, d, st, 32, 0, 0, out, 1, nullptr, 0, 0);
  togsim_dma(ctx, 1, 2, base, 2, d, st, 32, 0, 0, out, 1, nullptr, 0, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < 2; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, twostore_tile, iv, 1, i);
  }
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
