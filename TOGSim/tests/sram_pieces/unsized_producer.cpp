// A load into spad buffer 1 that the producer does not size: UNSIZED_CASE 0 states a table too
// short for it, 1 states no table at all. TOGSim refuses both instead of guessing from the DMAs.
#include "togsim_runtime.h"

#ifndef UNSIZED_CASE
#define UNSIZED_CASE 0
#endif

static void unsized_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t d[2] = {64, 64}, st[2] = {64, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  togsim_dma(ctx, 0, 0, 0, 2, d, st, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  int64_t iv[1] = {0};
  togsim_dispatch(ctx, unsized_tile, iv, 1, 0);
}

#if UNSIZED_CASE == 0
extern "C" const int32_t togsim_spad_buffer_count = 1;
extern "C" const int64_t togsim_spad_buffer_bytes[1] = {0};
#endif
