// A 16 KB output produced by a compute, then updated in place and stored on each of 4 trips: a
// work-item's spad footprint is 32 KB (input + state). Summing the stores made it 80 KB, more
// than the 48 KB spad, and the simulator never finished.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64
#define TRIPS 4

static void computestate_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t d[2] = {ROWS, COLS}, st[2] = {COLS, 1};
  const int64_t in[1] = {1}, s[1] = {2};
  togsim_dma(ctx, 0, 0, (uint64_t)iv[0] * ROWS * COLS, 2, d, st, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, s, 1);
  for (int k = 0; k < TRIPS; k++) {
    togsim_compute(ctx, 0, 0, 0, nullptr, s, 1, s, 1);
    togsim_dma(ctx, 1, 1, (uint64_t)(iv[0] * TRIPS + k) * ROWS * COLS, 2, d, st, 32, 0, 0, s, 1,
               nullptr, 0, 0);
  }
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < 2; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, computestate_tile, iv, 1, i);
  }
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
