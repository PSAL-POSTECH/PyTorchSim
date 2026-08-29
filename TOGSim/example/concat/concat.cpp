#include <cstddef>
#include <cstdint>
using std::size_t;

#include "togsim_runtime.h"

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

// config:  systolic_ws_8x8_c1_simple_noc_tpuv3.yml
// VLANE:   8
// mapping: autotune
static const int64_t A_ROWS = 128;
static const int64_t B_ROWS  = 128;
static const int64_t D    = 256;

static const int64_t TROW = 128;
static const int64_t TD   = 16;

static const int32_t ELEM_BITS = 32;

static const int64_t TILES_D = (D + TD - 1) / TD;

static const int32_t ARG_A = 0;
static const int32_t ARG_B   = 1;
static const int32_t ARG_OUT   = 2;

static const int64_t BUF_A[1] = {0};
static const int64_t BUF_B[1]   = {1};

static const int64_t TILE[2]   = {TROW, TD};
static const int64_t STRIDE[2] = {D, 1};

static const int32_t SYNC = 0;

static const int32_t TAG_A_IN  = 0;
static const int32_t TAG_A_OUT = 1;
static const int32_t TAG_B_IN    = 2;
static const int32_t TAG_B_OUT   = 3;

static void concat_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t d0 = iv[1];

  for (int64_t r = 0; r < A_ROWS; r += TROW) {
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_A, (uint64_t)(r * D + d0),
               2, TILE, STRIDE, ELEM_BITS,
               SYNC, TAG_A_IN, 0, nullptr, 0, BUF_A, 1);
    // MOVOUT
    togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(r * D + d0),
               2, TILE, STRIDE, ELEM_BITS,
               SYNC, TAG_A_OUT, 0, BUF_A, 1, nullptr, 0);
  }

  for (int64_t r = 0; r < B_ROWS; r += TROW) {
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_B, (uint64_t)(r * D + d0),
               2, TILE, STRIDE, ELEM_BITS,
               SYNC, TAG_B_IN, 0, nullptr, 0, BUF_B, 1);
    // MOVOUT
    togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)((r + A_ROWS) * D + d0),
               2, TILE, STRIDE, ELEM_BITS,
               SYNC, TAG_B_OUT, 0, BUF_B, 1, nullptr, 0);
  }
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;
  for (int64_t di = 0; di < TILES_D; ++di) {
    int64_t iv[2] = {0, di * TD};
    togsim_dispatch(ctx, concat_tile, iv, 2);
  }
}
