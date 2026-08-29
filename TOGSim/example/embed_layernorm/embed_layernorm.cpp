#include <cstddef>
#include <cstdint>
using std::size_t;

#include "togsim_runtime.h"

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

// config:  systolic_ws_8x8_c1_simple_noc_tpuv3.yml
// VLANE:   8
// mapping: autotune
static const int64_t SEQ = 128;
static const int64_t DIM = 256;

static const int64_t TSEQ = 128;
static const int64_t TDIM = 64;

static const int32_t ELEM_BITS = 32;
static const int32_t IDX_BITS  = 64;

static const int64_t TILES_DIM = (DIM + TDIM - 1) / TDIM;

static const int32_t ARG_IDS   = 0;
static const int32_t ARG_TABLE = 1;
static const int32_t ARG_OUT   = 2;

static const int64_t BUF_IDS[1] = {0};
static const int64_t BUF_ROW[1] = {1};

static const int64_t VEC_READ[2] = {0, 1};

static const int64_t TILE[2] = {TSEQ, TDIM};

static const int64_t STRIDE_IDS[2] = {1, 0};
static const int64_t STRIDE_ROW[2] = {0, 1};
static const int64_t STRIDE_OUT[2] = {DIM, 1};

static const int32_t SYNC = 0;

static const int32_t TAG_IDS   = 0;
static const int32_t TAG_TABLE = 1;
static const int32_t TAG_OUT   = 2;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_GATHER = 0;

static void embed_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t s0 = iv[0], d0 = iv[1];
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_IDS, (uint64_t)s0,
             2, TILE, STRIDE_IDS, IDX_BITS,
             SYNC, TAG_IDS, 0, nullptr, 0, BUF_IDS, 1);
  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_TABLE, (uint64_t)d0,
             2, TILE, STRIDE_ROW, ELEM_BITS,
             SYNC, TAG_TABLE, 0, BUF_IDS, 1, BUF_ROW, 1);

  // COMPUTE
  togsim_compute(ctx, TID_GATHER, CT_VECTOR, 0, nullptr,
                 VEC_READ, 2, nullptr, 0);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(s0 * DIM + d0),
             2, TILE, STRIDE_OUT, ELEM_BITS,
             SYNC, TAG_OUT, 0, BUF_ROW, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;
  for (int64_t s0 = 0; s0 < SEQ; s0 += TSEQ) {
    for (int64_t di = 0; di < TILES_DIM; ++di) {
      int64_t iv[2] = {s0, di * TDIM};
      togsim_dispatch(ctx, embed_tile, iv, 2);
    }
  }
}
