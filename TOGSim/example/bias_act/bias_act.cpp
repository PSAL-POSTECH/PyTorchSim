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

static const int32_t ARG_X    = 0;
static const int32_t ARG_BIAS = 1;
static const int32_t ARG_OUT  = 2;

static const int64_t BUF_BIAS[1] = {0};
static const int64_t BUF_X[1]    = {1};
static const int64_t BUF_OUT[1]  = {2};

static const int64_t VEC_READ[2] = {0, 1};

static const int64_t TILE[2] = {TM, TN};

static const int64_t STRIDE_2D[2]    = {N, 1};
static const int64_t STRIDE_BCAST[2] = {0, 1};

static const int32_t SYNC = 0;

static const int32_t TAG_X    = 0;
static const int32_t TAG_BIAS = 1;
static const int32_t TAG_OUT  = 2;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_BIAS_ACT = 0;

static void bias_gelu_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], n0 = iv[1];
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, TAG_X, 0, nullptr, 0, BUF_X, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_BIAS, (uint64_t)n0,
             2, TILE, STRIDE_BCAST, ELEM_BITS,
             SYNC, TAG_BIAS, 0, nullptr, 0, BUF_BIAS, 1);

  // COMPUTE
  togsim_compute(ctx, TID_BIAS_ACT, CT_VECTOR, 0, nullptr,
                 VEC_READ, 2, BUF_OUT, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, TAG_OUT, 0, BUF_OUT, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;
  for (int64_t mi = 0; mi < TILES_M; ++mi) {
    for (int64_t ni = 0; ni < TILES_N; ++ni) {
      int64_t iv[2] = {mi * TM, ni * TN};
      togsim_dispatch(ctx, bias_gelu_tile, iv, 2);
    }
  }
}
