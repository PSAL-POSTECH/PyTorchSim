// A 256 B index buffer loaded, a 16 KB output computed and scattered by one indirect store that
// reads both: the store drains only the output, so a work-item's spad footprint is 16640 B and
// two work-items share a 48 KB spad (sized by the store, the index buffer would count 16 KB).
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64

static void scatter_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t idx_dims[1] = {ROWS}, idx_st[1] = {1}, out_dims[2] = {ROWS, COLS}, out_st[2] = {COLS, 1};
  const int64_t idx[1] = {3}, out[1] = {2}, store_reads[2] = {2, 3};
  togsim_dma(ctx, 0, 0, (uint64_t)iv[0] * ROWS, 1, idx_dims, idx_st, 32, 0, 0, nullptr, 0, idx, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, idx, 1, out, 1);
  togsim_dma(ctx, 1, 1, 0, 2, out_dims, out_st, 32, 0, 0, store_reads, 2, nullptr, 0, 1);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < 2; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, scatter_tile, iv, 1, i);
  }
}
