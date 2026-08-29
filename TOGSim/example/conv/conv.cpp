#include <cstddef>
#include <cstdint>
using std::size_t;

#include "togsim_runtime.h"

int32_t togsim_abi_version(void) { return TOGSIM_ABI_VERSION; }

// config:  systolic_ws_8x8_c1_simple_noc_tpuv3.yml
// SA:      8x8 x2 per core
// VLANE:   8
// mapping: autotune
static const int64_t I_C = 64;
static const int64_t I_H = 16;
static const int64_t I_W = 16;
static const int64_t O_C = 64;
static const int64_t O_H = 16;
static const int64_t O_W = 16;
static const int64_t K_H = 3;
static const int64_t K_W = 3;

static const int64_t TILE_M = 16;
static const int64_t TILE_N = 64;
static const int64_t TILE_K = 64;
static const int64_t TILE_I_H = 8;

static const int64_t SA = 8;
static const int64_t VLANE = 8;
static const int32_t ELEM_BITS = 32;

static const int64_t PRE_ROWS = 8;
static const int64_t PRE_COLS = 8;
static const int64_t MM_ROWS  = 8;
static const int64_t MM_COLS  = 2;

static const int64_t W_SUBTILES = 8;
static const int64_t X_SUBTILES = 16;
static const int64_t ROW_PAIR = (I_W + 2) * I_C;
static const int64_t ROW_HALF = TILE_I_H * I_C;

static const int32_t ARG_X   = 0;
static const int32_t ARG_W   = 1;
static const int32_t ARG_OUT = 2;

static const int64_t BUF_ACC[1] = {0};
static const int64_t BUF_W[1]   = {1};
static const int64_t BUF_X[1]   = {2};
static const int64_t BUF_SA[1]  = {3};

static const int64_t PRELOAD_READ[2] = {1, 2};
static const int64_t MM_READ[2]      = {0, 3};

static const int64_t TILE_X[4]   = {1, 1, TILE_I_H, I_C};
static const int64_t TILE_W[4]   = {1, 1, O_C, TILE_I_H};
static const int64_t TILE_OUT[4] = {1, O_C, TILE_I_H, O_W};

static const int64_t STRIDE_X[4]   = {(I_H + 2) * (I_W + 2) * I_C, (I_W + 2) * I_C, I_C, 1};
static const int64_t STRIDE_W[4]   = {I_C * O_C * K_W, I_C * O_C, I_C, 1};
static const int64_t STRIDE_OUT[4] = {0, O_H * O_W, O_W, 1};

static const int32_t SYNC  = 0;
static const int32_t ASYNC = 1;

static const int32_t TAG_X   = 0;
static const int32_t TAG_W   = 1;
static const int32_t TAG_OUT = 2;

static const int32_t CT_VECTOR  = 0;
static const int32_t CT_MATMUL  = 1;
static const int32_t CT_PRELOAD = 2;

static const uint64_t TID_ACC_INIT = 0;
static const uint64_t TID_PRELOAD  = 1;
static const uint64_t TID_MATMUL   = 2;

static void conv_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t oc0 = iv[0], oh0 = iv[1], ow0 = iv[2];

  // COMPUTE
  togsim_compute(ctx, TID_ACC_INIT, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_ACC, 1);

  for (int64_t kh = 0; kh < K_H; ++kh) {
    for (int64_t kw = 0; kw < K_W; ++kw) {
      const int64_t pos = kh * K_W + kw;
      const int64_t x_off = (oh0 + kh) * (I_W + 2) * I_C + (ow0 + kw) * I_C;
      const int64_t w_off = oc0 * K_H * K_W * I_C;

      for (int64_t s = 0; s < W_SUBTILES; ++s) {
        // MOVIN
        togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_W, (uint64_t)(w_off + s * SA),
                   4, TILE_W, STRIDE_W, ELEM_BITS,
                   ASYNC, TAG_W, (uint64_t)s, nullptr, 0, BUF_W, 1);
      }
      for (int64_t s = 0; s < X_SUBTILES / 2; ++s) {
        // MOVIN
        togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X,
                   (uint64_t)(x_off + s * ROW_PAIR),
                   4, TILE_X, STRIDE_X, ELEM_BITS,
                   ASYNC, TAG_X, (uint64_t)(2 * s), nullptr, 0, BUF_X, 1);
        // MOVIN
        togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X,
                   (uint64_t)(x_off + s * ROW_PAIR + ROW_HALF),
                   4, TILE_X, STRIDE_X, ELEM_BITS,
                   ASYNC, TAG_X, (uint64_t)(2 * s + 1), nullptr, 0, BUF_X, 1);
      }

      for (int64_t pr = 0; pr < PRE_ROWS; ++pr) {
        for (int64_t pc = 0; pc < PRE_COLS; ++pc) {
          // MEMORY_BAR
          togsim_memory_barrier(ctx, TAG_W, (uint64_t)pc, BUF_W, 1);
          // COMPUTE
          togsim_compute(ctx, TID_PRELOAD, CT_PRELOAD, 0, nullptr,
                         PRELOAD_READ, 2, BUF_SA, 1);

          for (int64_t mr = 0; mr < MM_ROWS; ++mr) {
            for (int64_t mc = 0; mc < MM_COLS; ++mc) {
              // MEMORY_BAR
              togsim_memory_barrier(ctx, TAG_X, (uint64_t)(mr * MM_COLS + mc), BUF_X, 1);
              // COMPUTE
              togsim_compute(ctx, TID_MATMUL, CT_MATMUL, 0, nullptr,
                             MM_READ, 2, BUF_ACC, 1);
            }
          }
        }
      }
    }
  }

  // MOVOUT
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT,
             (uint64_t)(oc0 * O_H * O_W + oh0 * O_W + ow0),
             4, TILE_OUT, STRIDE_OUT, ELEM_BITS,
             SYNC, TAG_OUT, 0, BUF_ACC, 1, nullptr, 0);
}

// DISPATCH
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  (void)shape_args; (void)n;
  for (int64_t oc = 0; oc < O_C; oc += TILE_N) {
    for (int64_t oh = 0; oh < O_H; oh += TILE_I_H) {
      for (int64_t ow = 0; ow < O_W; ow += TILE_M) {
        int64_t iv[3] = {oc, oh, ow};
        togsim_dispatch(ctx, conv_tile, iv, 3);
      }
    }
  }
}
