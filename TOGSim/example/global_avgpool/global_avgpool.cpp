#include <cstddef>
#include <cstdint>
using std::size_t;

#include "togsim_runtime.h"

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

// config:  systolic_ws_8x8_c1_simple_noc_tpuv3.yml
// VLANE:   8
// mapping: autotune
static const int64_t C = 256;
static const int64_t HW = 64;

static const int64_t TC = 256;
static const int64_t THW = 64;
static const int64_t TC_SCALE = 128;

static const int32_t ELEM_BITS = 32;

static const int64_t TILES_C = (C + TC - 1) / TC;
static const int64_t TILES_C_SCALE = (C + TC_SCALE - 1) / TC_SCALE;

static const int32_t ARG_X   = 0;
static const int32_t ARG_SUM = 1;
static const int32_t ARG_OUT = 2;

static const int64_t BUF_X[1]   = {0};
static const int64_t BUF_SUM[1] = {1};
static const int64_t BUF_OUT[1] = {2};

static const int64_t TILE[2]       = {TC, THW};
static const int64_t TILE_ROW[1]   = {TC};
static const int64_t TILE_SCALE[1] = {TC_SCALE};

static const int64_t STRIDE_2D[2] = {HW, 1};
static const int64_t STRIDE_1D[1] = {1};

static const int32_t SYNC = 0;

static const int32_t TAG_X   = 0;
static const int32_t TAG_SUM = 1;
static const int32_t TAG_OUT = 2;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_REDUCE = 0;
static const uint64_t TID_WRITE  = 1;
static const uint64_t TID_SCALE  = 2;

static void sum_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t c0 = iv[0];

  for (int64_t hw0 = 0; hw0 < HW; hw0 += THW) {
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(c0 * HW + hw0),
               2, TILE, STRIDE_2D, ELEM_BITS,
               SYNC, TAG_X, 0, nullptr, 0, BUF_X, 1);
    // COMPUTE
    togsim_compute(ctx, TID_REDUCE, CT_VECTOR, 0, nullptr,
                   BUF_X, 1, nullptr, 0);
  }

  // COMPUTE
  togsim_compute(ctx, TID_WRITE, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_SUM, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_SUM, (uint64_t)c0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, TAG_SUM, 0, BUF_SUM, 1, nullptr, 0);
}

static void scale_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t c0 = iv[0];
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_SUM, (uint64_t)c0,
             1, TILE_SCALE, STRIDE_1D, ELEM_BITS,
             SYNC, TAG_SUM, 0, nullptr, 0, BUF_SUM, 1);

  // COMPUTE
  togsim_compute(ctx, TID_SCALE, CT_VECTOR, 0, nullptr,
                 BUF_SUM, 1, BUF_OUT, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)c0,
             1, TILE_SCALE, STRIDE_1D, ELEM_BITS,
             SYNC, TAG_OUT, 0, BUF_OUT, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;

  for (int64_t ci = 0; ci < TILES_C; ++ci) {
    int64_t iv[1] = {ci * TC};
    togsim_dispatch(ctx, sum_tile, iv, 1);
  }
  for (int64_t ci = 0; ci < TILES_C_SCALE; ++ci) {
    int64_t iv[1] = {ci * TC_SCALE};
    togsim_dispatch(ctx, scale_tile, iv, 1);
  }
}
