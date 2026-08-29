#include <cstddef>
#include <cstdint>
using std::size_t;

#include "togsim_runtime.h"

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

// config:  systolic_ws_8x8_c1_simple_noc_tpuv3.yml
// SA:      8x8 x2 per core
// VLANE:   8
// mapping: autotune
static const int64_t M = 128;
static const int64_t K = 256;
static const int64_t N = 256;

static const int64_t TM = 32;
static const int64_t TN = 256;
static const int64_t TK = 64;

static const int64_t SA = 8;
static const int64_t VLANE = 8;
static const int32_t ELEM_BITS = 32;

static const int64_t TILES_M = (M + TM - 1) / TM;
static const int64_t TILES_N = (N + TN - 1) / TN;

static const int64_t STEPS_M = TM / SA;
static const int64_t STEPS_N = TN / (SA * VLANE);

static const int32_t ARG_A = 0;
static const int32_t ARG_B = 1;
static const int32_t ARG_C = 2;

static const int64_t BUF_C[1]  = {0};
static const int64_t BUF_B[1]  = {1};
static const int64_t BUF_A[1]  = {2};
static const int64_t BUF_SA[1] = {3};

static const int64_t PRELOAD_READ[2] = {1, 2};
static const int64_t MM_READ[2]      = {0, 3};

static const int64_t TILE_A[2] = {TM, TK};
static const int64_t TILE_B[2] = {TK, TN};
static const int64_t TILE_C[2] = {TM, TN};
static const int64_t STRIDE[2] = {N, 1};

static const int32_t SYNC  = 0;
static const int32_t ASYNC = 1;

static const int32_t TAG_A = 0;
static const int32_t TAG_B = 1;
static const int32_t TAG_C = 2;

static const int32_t CT_VECTOR  = 0;
static const int32_t CT_MATMUL  = 1;
static const int32_t CT_PRELOAD = 2;

static const uint64_t TID_ACC_INIT = 0;
static const uint64_t TID_PRELOAD  = 1;
static const uint64_t TID_MATMUL   = 2;

static void gemm_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], n0 = iv[1];
  // COMPUTE
  togsim_compute(ctx, TID_ACC_INIT, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_C, 1);

  for (int64_t k0 = 0; k0 < K; k0 += TK) {
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_A, (uint64_t)(m0 * K + k0),
               2, TILE_A, STRIDE, ELEM_BITS,
               ASYNC, TAG_A, 0, nullptr, 0, BUF_A, 1);
    // MOVIN
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_B, (uint64_t)(k0 * N + n0),
               2, TILE_B, STRIDE, ELEM_BITS,
               ASYNC, TAG_B, 0, nullptr, 0, BUF_B, 1);

    for (int64_t k = 0; k < TK; ++k) {
      for (int64_t n = 0; n < STEPS_N; ++n) {
        // MEMORY_BAR
        togsim_memory_barrier(ctx, TAG_B, 0, BUF_B, 1);
        // COMPUTE
        togsim_compute(ctx, TID_PRELOAD, CT_PRELOAD, 0, nullptr,
                       PRELOAD_READ, 2, BUF_SA, 1);

        for (int64_t m = 0; m < STEPS_M; ++m) {
          // MEMORY_BAR
          togsim_memory_barrier(ctx, TAG_A, 0, BUF_A, 1);
          // COMPUTE
          togsim_compute(ctx, TID_MATMUL, CT_MATMUL, 0, nullptr,
                         MM_READ, 2, BUF_C, 1);
        }
      }
    }
  }

  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_C, (uint64_t)(m0 * N + n0),
             2, TILE_C, STRIDE, ELEM_BITS,
             SYNC, TAG_C, 0, BUF_C, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;
  for (int64_t mi = 0; mi < TILES_M; ++mi) {
    for (int64_t ni = 0; ni < TILES_N; ++ni) {
      int64_t iv[2] = {mi * TM, ni * TN};
      togsim_dispatch(ctx, gemm_tile, iv, 2);
    }
  }
}
