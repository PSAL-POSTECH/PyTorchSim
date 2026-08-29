#include <cstddef>
#include <cstdint>
using std::size_t;

#include "togsim_runtime.h"

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

// config:  systolic_ws_8x8_c1_simple_noc_tpuv3.yml
// VLANE:   8
// mapping: autotune
static const int64_t M = 128;
static const int64_t N = 256;

static const int64_t TM = 128;
static const int64_t TN = 64;

static const int32_t ELEM_BITS = 32;

static const int64_t TILES_M = (M + TM - 1) / TM;
static const int64_t TILES_N = (N + TN - 1) / TN;

static const int32_t ARG_X   = 0;
static const int32_t ARG_MAX = 1;
static const int32_t ARG_SUM = 2;
static const int32_t ARG_OUT = 3;

static const int64_t BUF_SUM[1] = {0};
static const int64_t BUF_X[1]   = {1};
static const int64_t BUF_MAX[1] = {2};
static const int64_t BUF_OUT[1] = {3};

static const int64_t READ_X_MAX[2]     = {1, 2};
static const int64_t READ_X_MAX_SUM[3] = {0, 1, 2};

static const int64_t TILE[2]     = {TM, TN};
static const int64_t TILE_ROW[1] = {TM};

static const int64_t STRIDE_2D[2]    = {N, 1};
static const int64_t STRIDE_BCAST[2] = {1, 0};
static const int64_t STRIDE_1D[1]    = {1};

static const int32_t SYNC = 0;

static const int32_t TAG_X   = 0;
static const int32_t TAG_MAX = 1;
static const int32_t TAG_SUM = 2;
static const int32_t TAG_OUT = 3;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_MAX_REDUCE = 0;
static const uint64_t TID_MAX_WRITE  = 1;
static const uint64_t TID_SUM_REDUCE = 2;
static const uint64_t TID_SUM_WRITE  = 3;
static const uint64_t TID_SOFTMAX    = 4;

static void max_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0];

  for (int64_t n0 = 0; n0 < N; n0 += TN) {
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
               2, TILE, STRIDE_2D, ELEM_BITS,
               SYNC, TAG_X, 0, nullptr, 0, BUF_X, 1);
    // COMPUTE
    togsim_compute(ctx, TID_MAX_REDUCE, CT_VECTOR, 0, nullptr,
                   BUF_X, 1, nullptr, 0);
  }

  // COMPUTE
  togsim_compute(ctx, TID_MAX_WRITE, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_MAX, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_MAX, (uint64_t)m0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, TAG_MAX, 0, BUF_MAX, 1, nullptr, 0);
}

static void sum_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0];

  for (int64_t n0 = 0; n0 < N; n0 += TN) {
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
               2, TILE, STRIDE_2D, ELEM_BITS,
               SYNC, TAG_X, 0, nullptr, 0, BUF_X, 1);
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_MAX, (uint64_t)m0,
               2, TILE, STRIDE_BCAST, ELEM_BITS,
               SYNC, TAG_MAX, 0, nullptr, 0, BUF_MAX, 1);
    // COMPUTE
    togsim_compute(ctx, TID_SUM_REDUCE, CT_VECTOR, 0, nullptr,
                   READ_X_MAX, 2, nullptr, 0);
  }

  // COMPUTE
  togsim_compute(ctx, TID_SUM_WRITE, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_SUM, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_SUM, (uint64_t)m0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, TAG_SUM, 0, BUF_SUM, 1, nullptr, 0);
}

static void softmax_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], n0 = iv[1];
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, TAG_X, 0, nullptr, 0, BUF_X, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_MAX, (uint64_t)m0,
             2, TILE, STRIDE_BCAST, ELEM_BITS,
             SYNC, TAG_MAX, 0, nullptr, 0, BUF_MAX, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_SUM, (uint64_t)m0,
             2, TILE, STRIDE_BCAST, ELEM_BITS,
             SYNC, TAG_SUM, 0, nullptr, 0, BUF_SUM, 1);

  // COMPUTE
  togsim_compute(ctx, TID_SOFTMAX, CT_VECTOR, 0, nullptr,
                 READ_X_MAX_SUM, 3, BUF_OUT, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, TAG_OUT, 0, BUF_OUT, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;

  for (int64_t mi = 0; mi < TILES_M; ++mi) {
    int64_t iv[1] = {mi * TM};
    togsim_dispatch(ctx, max_tile, iv, 1);
  }
  for (int64_t mi = 0; mi < TILES_M; ++mi) {
    int64_t iv[1] = {mi * TM};
    togsim_dispatch(ctx, sum_tile, iv, 1);
  }
  for (int64_t mi = 0; mi < TILES_M; ++mi) {
    for (int64_t ni = 0; ni < TILES_N; ++ni) {
      int64_t iv[2] = {mi * TM, ni * TN};
      togsim_dispatch(ctx, softmax_tile, iv, 2);
    }
  }
}
