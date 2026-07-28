#include "Extra.h"

namespace at::native::openreg {

at::Tensor quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype) {
  return at::native::quantize_per_tensor(self, scale, zero_point, dtype);
}

int64_t _fused_sdp_choice(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {

  sdp::sdp_params params{query, key, value, attn_mask, dropout_p, is_causal, enable_gqa};

  // Reject inputs that are fundamentally unsupported (e.g. wrong rank)
  if (!sdp::check_tensor_shapes(params, /*debug=*/false)) {
    return static_cast<int64_t>(sdp::SDPBackend::error);
  }

  // q: (B, Hq, L, E)   k/v: (B, H, S, E)
  const int64_t Hq = query.size(-3);
  const int64_t H  = key.size(-3);
  const int64_t L  = query.size(-2);  // query sequence length
  const int64_t S  = key.size(-2);    // key/value sequence length

  // Conditions required by the MLIR FlashSDPA kernel:
  // Prefill only  : L == S  (decode has L == 1, not supported)
  // Non-GQA       : Hq == H (equal query and KV heads)
  // No dropout    : template has no dropout implementation
  // Dense tensors : no nested tensor support
  const bool can_use_mlir_flash =
      (L == S) &&
      (Hq == H) && !enable_gqa &&
      sdp::check_for_dropout(params, /*debug=*/false) &&
      sdp::check_nested_tensor(params, /*debug=*/false);

  const bool ctx_flash        = at::globalContext().userEnabledFlashSDP();
  const bool ctx_math         = at::globalContext().userEnabledMathSDP();

  if (ctx_flash && can_use_mlir_flash) {
    return static_cast<int64_t>(sdp::SDPBackend::overrideable);
  }

  return static_cast<int64_t>(sdp::SDPBackend::math);
}

void quantize_tensor_per_tensor_affine_stub(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale) {
  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
      at::empty_like(query),
      at::empty_like(key),
      at::empty_like(value),
      at::empty_like(attn_bias));
}

namespace {
struct CustomAutogradFnReturnsSelf
    : public torch::autograd::Function<CustomAutogradFnReturnsSelf> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

struct CustomAutogradFnAliasing
    : public torch::autograd::Function<CustomAutogradFnAliasing> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self.view_symint(self.sym_sizes());
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};
} // namespace

at::Tensor custom_autograd_fn_returns_self(at::Tensor x) {
  return CustomAutogradFnReturnsSelf::apply(x);
}

at::Tensor custom_autograd_fn_aliasing(at::Tensor x) {
  return CustomAutogradFnAliasing::apply(x);
}

/*
 This implementation is only used to test stub registration, so not all
 capabilities are fully supported.

 Current Limitations:
 - dtype: Float only
 - input tensor: must be contiguous layout
*/
// LITERALINCLUDE START: STUB ABS
void abs_kernel(at::TensorIteratorBase& iter) {
  TORCH_CHECK(iter.ntensors() == 2, "Abs kernel expects 2 tensors");
  TORCH_CHECK(
      iter.common_dtype() == at::ScalarType::Float,
      "Abs kernel only supports float type");

  auto& output_tensor = iter.tensor(0);
  auto& input_tensor = iter.tensor(1);

  TORCH_CHECK(
      input_tensor.sizes() == output_tensor.sizes(),
      "Input and output tensor sizes must match.");

  auto abs_loop = [](float* out_ptr, const float* in_ptr, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      out_ptr[i] = std::abs(in_ptr[i]);
    }
  };

  MemoryGuard guard(input_tensor, output_tensor);

  if (iter.is_contiguous()) {
    abs_loop(
        static_cast<float*>(iter.data_ptr(0)),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());
  } else {
    TORCH_CHECK(
        input_tensor.is_contiguous(), "Input tensor must be contiguous.")

    auto output = at::empty(
        input_tensor.sizes(),
        input_tensor.options().memory_format(
            input_tensor.suggest_memory_format()));

    MemoryGuard guard(output);

    abs_loop(
        static_cast<float*>(output.data_ptr()),
        static_cast<float*>(iter.data_ptr(1)),
        iter.numel());

    output_tensor.copy_(output);
  }
}
// LITERALINCLUDE END: STUB ABS

at::Tensor& abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::abs_out(self, out);
}

at::Tensor custom_abs(at::Tensor x) {
  return at::abs(x);
}

} // namespace at::native::openreg
