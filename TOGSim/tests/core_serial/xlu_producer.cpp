// Two work-items; in each, one load feeds a cross-lane compute (tile 0) and a vector compute (tile 1)
// that do not depend on each other: they ran side by side on the XLU and VPU; a core runs one at a time.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64

static void xlu_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t d[2] = {ROWS, COLS}, st[2] = {COLS, 1};
  const int64_t in[1] = {1}, x[1] = {2}, y[1] = {3};
  togsim_dma(ctx, 0, 0, 0, 2, d, st, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 3, 0, nullptr, in, 1, x, 1);
  togsim_compute(ctx, 1, 0, 0, nullptr, in, 1, y, 1);
  togsim_dma(ctx, 1, 1, 0, 2, d, st, 32, 0, 0, x, 1, nullptr, 0, 0);
  togsim_dma(ctx, 1, 2, 0, 2, d, st, 32, 0, 0, y, 1, nullptr, 0, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < 2; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, xlu_tile, iv, 1, i);
  }
}

extern "C" const int32_t togsim_spad_buffer_count = 4;
extern "C" const int64_t togsim_spad_buffer_bytes[4] = {0, ROWS * COLS * 4, ROWS * COLS * 4, ROWS * COLS * 4};
