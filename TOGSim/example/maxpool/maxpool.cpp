#include <cstddef>
#include <cstdint>
using std::size_t;

#include "togsim_runtime.h"

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

// config:  systolic_ws_8x8_c1_simple_noc_tpuv3.yml
// VLANE:   8
// mapping: autotune
static const int64_t ROWS = 1024;
static const int64_t COLS = 16;

static const int64_t TROW = 256;
static const int64_t TCOL = 16;

static const int32_t ELEM_BITS = 32;

static const int64_t TILES_ROW = (ROWS + TROW - 1) / TROW;
static const int64_t TILES_COL = (COLS + TCOL - 1) / TCOL;

static const int64_t IN_ROW_STRIDE = 64;
static const int64_t IN_W          = 32;

static const int32_t ARG_X   = 0;
static const int32_t ARG_OUT = 1;

static const int64_t BUF_W3[1]  = {0};
static const int64_t BUF_W0[1]  = {1};
static const int64_t BUF_W1[1]  = {2};
static const int64_t BUF_W2[1]  = {3};
static const int64_t BUF_OUT[1] = {4};

static const int64_t VEC_READ[4] = {0, 1, 2, 3};

static const int64_t TILE[2] = {TROW, TCOL};

static const int64_t STRIDE_IN[2]  = {IN_ROW_STRIDE, 2};
static const int64_t STRIDE_OUT[2] = {TCOL, 1};

static const int32_t SYNC = 0;

static const int32_t TAG_W0  = 0;
static const int32_t TAG_W1  = 1;
static const int32_t TAG_W2  = 2;
static const int32_t TAG_W3  = 3;
static const int32_t TAG_OUT = 4;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_MAX = 0;

static void maxpool_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t r0 = iv[0], c0 = iv[1];
  const int64_t base = r0 * IN_ROW_STRIDE + c0 * 2;
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)base,
             2, TILE, STRIDE_IN, ELEM_BITS,
             SYNC, TAG_W0, 0, nullptr, 0, BUF_W0, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(base + 1),
             2, TILE, STRIDE_IN, ELEM_BITS,
             SYNC, TAG_W1, 0, nullptr, 0, BUF_W1, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(base + IN_W),
             2, TILE, STRIDE_IN, ELEM_BITS,
             SYNC, TAG_W2, 0, nullptr, 0, BUF_W2, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(base + IN_W + 1),
             2, TILE, STRIDE_IN, ELEM_BITS,
             SYNC, TAG_W3, 0, nullptr, 0, BUF_W3, 1);
  // COMPUTE
  togsim_compute(ctx, TID_MAX, CT_VECTOR, 0, nullptr,
                 VEC_READ, 4, BUF_OUT, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(r0 * TCOL + c0),
             2, TILE, STRIDE_OUT, ELEM_BITS,
             SYNC, TAG_OUT, 0, BUF_OUT, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;
  for (int64_t ri = 0; ri < TILES_ROW; ++ri) {
    for (int64_t ci = 0; ci < TILES_COL; ++ci) {
      int64_t iv[2] = {ri * TROW, ci * TCOL};
      togsim_dispatch(ctx, maxpool_tile, iv, 2);
    }
  }
}
