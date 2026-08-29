#include <cstddef>
#include <cstdint>
using std::size_t;

#include "togsim_runtime.h"

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

// config:  systolic_ws_8x8_c1_simple_noc_tpuv3.yml
// VLANE:   8
// mapping: autotune
static const int64_t TOKENS = 128;
static const int64_t HIDDEN = 256;
static const int64_t NUM_HEADS = 8;
static const int64_t NUM_KV_HEADS = 8;
static const int64_t PAST = 128;

static const int64_t CACHE_DIM = HIDDEN / NUM_HEADS * NUM_KV_HEADS;
static const int64_t QKV_WIDTH = HIDDEN + 2 * CACHE_DIM;

static const int64_t TTOK = 32;

static const int32_t ELEM_BITS = 32;

static const int64_t TILES_TOK = (TOKENS + TTOK - 1) / TTOK;

static const int32_t ARG_QKV   = 0;
static const int32_t ARG_QUERY = 1;
static const int32_t ARG_KEY   = 2;
static const int32_t ARG_VALUE = 3;

static const int64_t BUF_QKV[1]   = {0};
static const int64_t BUF_QUERY[1] = {1};
static const int64_t BUF_KEY[1]   = {2};
static const int64_t BUF_VALUE[1] = {3};

static const int64_t SPLIT_WRITE[3] = {1, 2, 3};

static const int64_t TILE_QKV[2]   = {TTOK, QKV_WIDTH};
static const int64_t TILE_QUERY[2] = {TTOK, HIDDEN};
static const int64_t TILE_CACHE[2] = {TTOK, CACHE_DIM};

static const int64_t STRIDE_QKV[2]   = {QKV_WIDTH, 1};
static const int64_t STRIDE_QUERY[2] = {HIDDEN, 1};
static const int64_t STRIDE_CACHE[2] = {CACHE_DIM, 1};

static const int32_t SYNC = 0;

static const int32_t TAG_QKV   = 0;
static const int32_t TAG_QUERY = 1;
static const int32_t TAG_KEY   = 2;
static const int32_t TAG_VALUE = 3;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_SPLIT = 0;

static void split_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t t0 = iv[0];

  // MOVIN
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_QKV, (uint64_t)(t0 * QKV_WIDTH),
             2, TILE_QKV, STRIDE_QKV, ELEM_BITS,
             SYNC, TAG_QKV, 0, nullptr, 0, BUF_QKV, 1);

  // COMPUTE
  togsim_compute(ctx, TID_SPLIT, CT_VECTOR, 0, nullptr,
                 BUF_QKV, 1, SPLIT_WRITE, 3);

  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_QUERY, (uint64_t)(t0 * HIDDEN),
             2, TILE_QUERY, STRIDE_QUERY, ELEM_BITS,
             SYNC, TAG_QUERY, 0, BUF_QUERY, 1, nullptr, 0);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_KEY,
             (uint64_t)((PAST + t0) * CACHE_DIM),
             2, TILE_CACHE, STRIDE_CACHE, ELEM_BITS,
             SYNC, TAG_KEY, 0, BUF_KEY, 1, nullptr, 0);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_VALUE,
             (uint64_t)((PAST + t0) * CACHE_DIM),
             2, TILE_CACHE, STRIDE_CACHE, ELEM_BITS,
             SYNC, TAG_VALUE, 0, BUF_VALUE, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;
  for (int64_t ti = 0; ti < TILES_TOK; ++ti) {
    int64_t iv[1] = {ti * TTOK};
    togsim_dispatch(ctx, split_tile, iv, 1);
  }
}
