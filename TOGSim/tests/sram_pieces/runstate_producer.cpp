// A 16 KB state loaded once, updated in place and stored on each of 4 trips: one buffer, so a
// work-item's spad footprint is 16 KB (not 4 x 16 KB of stores) and two work-items share the spad.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64
#define TRIPS 4

static void runstate_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t d[2] = {ROWS, COLS}, st[2] = {COLS, 1};
  const int64_t s[1] = {1};
  togsim_dma(ctx, 0, 0, (uint64_t)iv[0] * ROWS * COLS, 2, d, st, 32, 0, 0, nullptr, 0, s, 1, 0);
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
    togsim_dispatch(ctx, runstate_tile, iv, 1, i);
  }
}

extern "C" const int32_t togsim_spad_buffer_count = 2;
extern "C" const int64_t togsim_spad_buffer_bytes[2] = {0, ROWS * COLS * 4};
