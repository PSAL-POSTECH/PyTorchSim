#include <cstddef>
#include <cstdint>
using std::size_t;

#include "togsim_runtime.h"

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

// config:  systolic_ws_8x8_c1_simple_noc_tpuv3.yml
// SA:      8x8 x2 per core
// VLANE:   8
// mapping: autotune
static const int64_t HEADS = 8;
static const int64_t KV_HEADS = 8;
static const int64_t SEQ = 128;
static const int64_t DHEAD = 64;

static const int64_t Q_LEN = 8;
static const int64_t KV_BLOCK = 64;

static const int64_t SA = 8;
static const int64_t HEADS_PER_KV = HEADS / KV_HEADS;

static const int32_t ELEM_BITS = 32;

static const int64_t TILES_Q = (SEQ + Q_LEN - 1) / Q_LEN;
static const int64_t TILES_KV = (SEQ + KV_BLOCK - 1) / KV_BLOCK;

static const int64_t QK_S_STEPS = (KV_BLOCK + SA - 1) / SA;
static const int64_t QK_K_STEPS = (DHEAD + SA - 1) / SA;
static const int64_t SV_K_STEPS = (DHEAD + SA - 1) / SA;
static const int64_t SV_S_STEPS = (KV_BLOCK + SA - 1) / SA;

static const int32_t ARG_Q   = 0;
static const int32_t ARG_K   = 1;
static const int32_t ARG_V   = 2;
static const int32_t ARG_OUT = 3;

static const int64_t BUF_K[1]     = {0};
static const int64_t BUF_Q[1]     = {1};
static const int64_t BUF_V[1]     = {2};
static const int64_t BUF_SA_QK[1] = {3};
static const int64_t BUF_LOGIT[1] = {4};
static const int64_t BUF_STAT[1]  = {5};
static const int64_t BUF_ACC[1]   = {6};
static const int64_t BUF_SA_SV[1] = {7};

static const int64_t QK_PRELOAD_READ[2] = {0, 1};
static const int64_t QK_MM_READ[2]      = {3, 4};
static const int64_t SV_PRELOAD_READ[2] = {2, 4};
static const int64_t SV_MM_READ[2]      = {6, 7};

static const int64_t STAT_READ[2]  = {4, 5};
static const int64_t LOGIT_READ[2] = {4, 5};
static const int64_t ACC_READ[2]   = {5, 6};

static const int64_t TILE_Q[2]   = {Q_LEN, DHEAD};
static const int64_t TILE_KV[2]  = {KV_BLOCK, DHEAD};
static const int64_t TILE_OUT[2] = {Q_LEN, DHEAD};

static const int64_t STRIDE_Q[2]   = {DHEAD, 1};
static const int64_t STRIDE_KV[2]  = {DHEAD, 1};
static const int64_t STRIDE_OUT[2] = {DHEAD, 1};

static const int32_t SYNC  = 0;
static const int32_t ASYNC = 1;

static const int32_t TAG_K = 0;
static const int32_t TAG_Q = 1;
static const int32_t TAG_V = 2;
static const int32_t TAG_O = 3;

static const int32_t CT_VECTOR  = 0;
static const int32_t CT_MATMUL  = 1;
static const int32_t CT_PRELOAD = 2;

static const uint64_t TID_QK_PRELOAD = 0;
static const uint64_t TID_QK_MATMUL  = 1;
static const uint64_t TID_ROWMAX     = 2;
static const uint64_t TID_SUB        = 3;
static const uint64_t TID_EXP        = 4;
static const uint64_t TID_ROWSUM     = 5;
static const uint64_t TID_MAC        = 6;
static const uint64_t TID_RESCALE    = 7;
static const uint64_t TID_SV_PRELOAD = 8;
static const uint64_t TID_SV_MATMUL  = 9;
static const uint64_t TID_NORM       = 10;

static void attention_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t h = iv[0], q0 = iv[1];

  for (int64_t kv0 = 0; kv0 < SEQ; kv0 += KV_BLOCK) {
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_K,
               (uint64_t)(h * SEQ * DHEAD + kv0 * DHEAD),
               2, TILE_KV, STRIDE_KV, ELEM_BITS,
               ASYNC, TAG_K, 0, nullptr, 0, BUF_K, 1);
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_Q,
               (uint64_t)(h * SEQ * DHEAD + q0 * DHEAD),
               2, TILE_Q, STRIDE_Q, ELEM_BITS,
               ASYNC, TAG_Q, 0, nullptr, 0, BUF_Q, 1);
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_V,
               (uint64_t)(h * SEQ * DHEAD + kv0 * DHEAD),
               2, TILE_KV, STRIDE_KV, ELEM_BITS,
               ASYNC, TAG_V, 0, nullptr, 0, BUF_V, 1);

    for (int64_t s = 0; s < QK_S_STEPS; ++s) {
      for (int64_t k = 0; k < QK_K_STEPS; ++k) {
        // MEMORY_BAR
        togsim_memory_barrier(ctx, TAG_K, 0, BUF_K, 1);
        // COMPUTE
        togsim_compute(ctx, TID_QK_PRELOAD, CT_PRELOAD, 0, nullptr,
                       QK_PRELOAD_READ, 2, BUF_SA_QK, 1);
        for (int64_t hh = 0; hh < HEADS_PER_KV; ++hh) {
          // MEMORY_BAR
          togsim_memory_barrier(ctx, TAG_Q, 0, BUF_Q, 1);
          // COMPUTE
          togsim_compute(ctx, TID_QK_MATMUL, CT_MATMUL, 0, nullptr,
                         QK_MM_READ, 2, BUF_LOGIT, 1);
        }
      }
    }

    // COMPUTE
    togsim_compute(ctx, TID_ROWMAX, CT_VECTOR, 0, nullptr,
                   BUF_LOGIT, 1, BUF_STAT, 1);
    // COMPUTE
    togsim_compute(ctx, TID_SUB, CT_VECTOR, 0, nullptr,
                   STAT_READ, 2, BUF_LOGIT, 1);
    // COMPUTE
    togsim_compute(ctx, TID_EXP, CT_VECTOR, 0, nullptr,
                   BUF_LOGIT, 1, BUF_LOGIT, 1);
    // COMPUTE
    togsim_compute(ctx, TID_ROWSUM, CT_VECTOR, 0, nullptr,
                   BUF_LOGIT, 1, BUF_STAT, 1);
    // COMPUTE
    togsim_compute(ctx, TID_MAC, CT_VECTOR, 0, nullptr,
                   LOGIT_READ, 2, BUF_STAT, 1);
    // COMPUTE
    togsim_compute(ctx, TID_RESCALE, CT_VECTOR, 0, nullptr,
                   ACC_READ, 2, BUF_ACC, 1);

    for (int64_t k = 0; k < SV_K_STEPS; ++k) {
      for (int64_t s = 0; s < SV_S_STEPS; ++s) {
        // MEMORY_BAR
        togsim_memory_barrier(ctx, TAG_V, 0, BUF_V, 1);
        // COMPUTE
        togsim_compute(ctx, TID_SV_PRELOAD, CT_PRELOAD, 0, nullptr,
                       SV_PRELOAD_READ, 2, BUF_SA_SV, 1);
        for (int64_t hh = 0; hh < HEADS_PER_KV; ++hh) {
          // COMPUTE
          togsim_compute(ctx, TID_SV_MATMUL, CT_MATMUL, 0, nullptr,
                         SV_MM_READ, 2, BUF_ACC, 1);
        }
      }
    }
  }

  // COMPUTE
  togsim_compute(ctx, TID_NORM, CT_VECTOR, 0, nullptr,
                 ACC_READ, 2, BUF_ACC, 1);
  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT,
             (uint64_t)(h * SEQ * DHEAD + q0 * DHEAD),
             2, TILE_OUT, STRIDE_OUT, ELEM_BITS,
             SYNC, TAG_O, 0, BUF_ACC, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;
  for (int64_t h = 0; h < KV_HEADS; ++h) {
    for (int64_t qi = 0; qi < TILES_Q; ++qi) {
      int64_t iv[2] = {h, qi * Q_LEN};
      togsim_dispatch(ctx, attention_tile, iv, 2);
    }
  }
}
