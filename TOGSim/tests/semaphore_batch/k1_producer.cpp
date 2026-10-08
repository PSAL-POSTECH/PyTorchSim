// K1: two async loads signal one semaphore; the first is strided across DRAM rows so the
// second, issued later, arrives first. One barrier waits both, one compute reads the buffer.
// K1_CASE 1/2/3 break the contract (wrong count / never-signaled wait / no wait).
#include "togsim_runtime.h"

#ifndef K1_CASE
#define K1_CASE 0
#endif
#define K1_ROWS 256
#define K1_ROW_STRIDE 65536

static void k1_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t big[2] = {K1_ROWS, 16}, small[2] = {1, 16};
  const int64_t big_st[2] = {K1_ROW_STRIDE, 1}, small_st[2] = {16, 1};
  const int64_t buf[1] = {1}, out[1] = {2};
#if TOGSIM_ABI_VERSION >= 14
  togsim_dma(ctx, 0, 0, 0, 2, big, big_st, 32, 1, 7, nullptr, 0, buf, 1, 0);
  togsim_dma(ctx, 0, 1, 0, 2, small, small_st, 32, 1, 7, nullptr, 0, buf, 1, 0);
#if K1_CASE == 0
  togsim_memory_barrier(ctx, 7, K1_ROWS * 16 + 16);
#elif K1_CASE == 1
  togsim_memory_barrier(ctx, 7, 16);
#elif K1_CASE == 2
  togsim_memory_barrier(ctx, 8, 16);
#endif
#else
  togsim_dma(ctx, 0, 0, 0, 2, big, big_st, 32, 1, 0, 7, nullptr, 0, buf, 1, 0);
  togsim_dma(ctx, 0, 1, 0, 2, small, small_st, 32, 1, 0, 7, nullptr, 0, buf, 1, 0);
  togsim_memory_barrier(ctx, 0, 7, buf, 1);
#endif
  togsim_compute(ctx, 0, 0, 0, nullptr, buf, 1, out, 1);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  int64_t iv[1] = {0};
  togsim_dispatch(ctx, k1_tile, iv, 1, 0);
}
