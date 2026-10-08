// Re-wait: load A on sem 3, wait, A = f(A) in place on the cross-lane unit (tile 0, long),
// wait sem 3 again, then a vector compute (tile 1, short) reads A: it must follow f, not the
// barrier. REWAIT_CASE 1 re-waits expecting another count.
#include "togsim_runtime.h"

#ifndef REWAIT_CASE
#define REWAIT_CASE 0
#endif

static void rewait_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[2] = {8, 16}, st[2] = {16, 1};
  const int64_t a[1] = {1}, out[1] = {2};
  togsim_dma(ctx, 0, 0, 0, 2, dims, st, 32, 1, 3, nullptr, 0, a, 1, 0);
  togsim_memory_barrier(ctx, 3, 128);
  togsim_compute(ctx, 0, 3, 0, nullptr, a, 1, a, 1);
  togsim_memory_barrier(ctx, 3, REWAIT_CASE == 1 ? 64 : 128);
  togsim_compute(ctx, 1, 0, 0, nullptr, a, 1, out, 1);
  togsim_dma(ctx, 1, 1, 0, 2, dims, st, 32, 0, 0, out, 1, nullptr, 0, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  int64_t iv[1] = {0};
  togsim_dispatch(ctx, rewait_tile, iv, 1, 0);
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, 8 * 16 * 4, 8 * 16 * 4};
