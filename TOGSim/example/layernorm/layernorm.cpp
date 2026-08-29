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

static const int64_t TM = 64;
static const int64_t TN = 64;
static const int64_t TM_STATS = 128;

static const int32_t ELEM_BITS = 32;

static const int64_t TILES_M = (M + TM - 1) / TM;
static const int64_t TILES_N = (N + TN - 1) / TN;
static const int64_t TILES_M_STATS = (M + TM_STATS - 1) / TM_STATS;

static const int32_t ARG_X    = 0;
static const int32_t ARG_SKIP = 1;
static const int32_t ARG_MEAN = 2;
static const int32_t ARG_VAR  = 3;
static const int32_t ARG_W    = 4;
static const int32_t ARG_B    = 5;
static const int32_t ARG_OUT  = 6;

static const int64_t BUF_X[1]    = {0};
static const int64_t BUF_W[1]    = {1};
static const int64_t BUF_VAR[1]  = {2};
static const int64_t BUF_MEAN[1] = {3};
static const int64_t BUF_SKIP[1] = {4};
static const int64_t BUF_B[1]    = {5};
static const int64_t BUF_OUT[1]  = {6};

static const int64_t READ_X_SKIP[2] = {0, 4};
static const int64_t READ_ALL[6]    = {0, 1, 2, 3, 4, 5};

static const int64_t TILE[2]       = {TM, TN};
static const int64_t TILE_STATS[2] = {TM_STATS, TN};
static const int64_t TILE_ROW[1]   = {TM_STATS};

static const int64_t STRIDE_2D[2]  = {N, 1};
static const int64_t STRIDE_1D[1]  = {1};
static const int64_t STRIDE_ROW[2] = {1, 0};
static const int64_t STRIDE_COL[2] = {0, 1};

static const int32_t SYNC = 0;

static const int32_t TAG_X    = 0;
static const int32_t TAG_SKIP = 1;
static const int32_t TAG_MEAN = 2;
static const int32_t TAG_VAR  = 3;
static const int32_t TAG_W    = 4;
static const int32_t TAG_B    = 5;
static const int32_t TAG_OUT  = 6;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_REDUCE     = 0;
static const uint64_t TID_WRITE_MEAN = 1;
static const uint64_t TID_WRITE_VAR  = 2;
static const uint64_t TID_NORM       = 3;

static void stats_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0];

  for (int64_t n0 = 0; n0 < N; n0 += TN) {
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
               2, TILE_STATS, STRIDE_2D, ELEM_BITS,
               SYNC, TAG_X, 0, nullptr, 0, BUF_X, 1);
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_SKIP, (uint64_t)(m0 * N + n0),
               2, TILE_STATS, STRIDE_2D, ELEM_BITS,
               SYNC, TAG_SKIP, 0, nullptr, 0, BUF_SKIP, 1);
    // COMPUTE
    togsim_compute(ctx, TID_REDUCE, CT_VECTOR, 0, nullptr,
                   READ_X_SKIP, 2, nullptr, 0);
  }

  // COMPUTE
  togsim_compute(ctx, TID_WRITE_MEAN, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_MEAN, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_MEAN, (uint64_t)m0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, TAG_MEAN, 0, BUF_MEAN, 1, nullptr, 0);
  // COMPUTE
  togsim_compute(ctx, TID_WRITE_VAR, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_VAR, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_VAR, (uint64_t)m0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, TAG_VAR, 0, BUF_VAR, 1, nullptr, 0);
}

static void norm_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], n0 = iv[1];
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, TAG_X, 0, nullptr, 0, BUF_X, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_SKIP, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, TAG_SKIP, 0, nullptr, 0, BUF_SKIP, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_MEAN, (uint64_t)m0,
             2, TILE, STRIDE_ROW, ELEM_BITS,
             SYNC, TAG_MEAN, 0, nullptr, 0, BUF_MEAN, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_VAR, (uint64_t)m0,
             2, TILE, STRIDE_ROW, ELEM_BITS,
             SYNC, TAG_VAR, 0, nullptr, 0, BUF_VAR, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_W, (uint64_t)n0,
             2, TILE, STRIDE_COL, ELEM_BITS,
             SYNC, TAG_W, 0, nullptr, 0, BUF_W, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_B, (uint64_t)n0,
             2, TILE, STRIDE_COL, ELEM_BITS,
             SYNC, TAG_B, 0, nullptr, 0, BUF_B, 1);

  // COMPUTE
  togsim_compute(ctx, TID_NORM, CT_VECTOR, 0, nullptr,
                 READ_ALL, 6, BUF_OUT, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, TAG_OUT, 0, BUF_OUT, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;

  for (int64_t mi = 0; mi < TILES_M_STATS; ++mi) {
    int64_t iv[1] = {mi * TM_STATS};
    togsim_dispatch(ctx, stats_tile, iv, 1);
  }
  for (int64_t mi = 0; mi < TILES_M; ++mi) {
    for (int64_t ni = 0; ni < TILES_N; ++ni) {
      int64_t iv[2] = {mi * TM, ni * TN};
      togsim_dispatch(ctx, norm_tile, iv, 2);
    }
  }
}
