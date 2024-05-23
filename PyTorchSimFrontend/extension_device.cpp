#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/MemoryFormat.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/CPUFallback.h>

static uint64_t op_counter = 0;
static uint64_t last_saved_value = 0;

// register guard
namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);

}} // namespace at::detail

// basic dummy add function
at::Tensor custom_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  op_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

// basic dummy mul function
at::Tensor custom_mul_Tensor(const at::Tensor & self, const at::Tensor & other) {
  op_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

at::Tensor _reinterpret_tensor(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    int64_t offset_increment) {
  at::Tensor self_ = at::detail::make_tensor<c10::TensorImpl>(
      c10::Storage(self.storage()), self.key_set(), self.dtype());
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(self.storage_offset() + offset_increment);
  self_tmp_->set_sizes_and_strides(size, stride);
  return self_;
}

at::Tensor& zero_inplace_batching_rule(at::Tensor &self) {
  op_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return self;
}

const at::Tensor& custom_resize_(const at::Tensor& self, at::IntArrayRef size,
                          std::optional<at::MemoryFormat> optional_memory_format) {
  at::TensorImpl* tensor_impl = self.unsafeGetTensorImpl();
  tensor_impl->set_sizes_contiguous(size);
  const auto itemsize = tensor_impl->dtype().itemsize();
  const auto offset = tensor_impl->storage_offset();
  const auto storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize, offset);
  // Dummy device is using cpu allocator, so here just call cpu
  // function maybe_resize_storage_cpu in aten/src/ATen/native/Resize.h
  // to get a sufficient memory space.
  at::native::maybe_resize_storage_cpu(tensor_impl, storage_size);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != at::MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    tensor_impl->empty_tensor_restride(memory_format);
  }
  return self;
}

// basic dummy eq function: Only support CPU
at::Tensor custom_to_device(
    const at::Tensor & self,
    at::Device device,
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    c10::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(device.is_cpu() || device.type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  TORCH_CHECK(self.scalar_type() == dtype);
  TORCH_CHECK(self.is_contiguous());

  op_counter += 1;
  if (device != at::DeviceType::CPU) {
    return at::empty(self.sizes(), self.options());
  }

  auto out = at::empty(self.sizes(), dtype, self.options().layout(), device, false, memory_format);
  memcpy(out.mutable_data_ptr(), self.mutable_data_ptr(), self.nbytes());
  // Since this custom device is just for testing, not bothering to implement kernels.
  return out;
}


// A dummy allocator for our custom device, that secretly uses the CPU
struct DummyCustomAllocator final : at::Allocator {
  DummyCustomAllocator() = default;
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = c10::alloc_cpu(nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    c10::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }
};

// Register our dummy allocator
static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows dummy device.");
  TORCH_CHECK(self.is_contiguous());
  TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);

  op_counter += 1;
  auto _data = static_cast<float*>(self.mutable_data_ptr());
  for (size_t idx = 0; idx < self.numel(); idx++) {
    _data[idx] = value.toFloat();
  }

  return self;
}

at::Tensor unsafe_create_cpu_tensor_from_dummy_tensor(const at::Tensor& src) {
  // TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
  //             "Only support dummy device.");
  const auto& sizes_ = src.sizes();
  const auto& strides_ = src.strides();
  auto storage_offset_ = src.storage_offset();
  at::detail::check_size_nonnegative(sizes_);

  size_t size_bytes = at::detail::computeStorageNbytes(sizes_, strides_,
                                                       src.element_size(),
                                                       storage_offset_);

  at::DataPtr data_ptr =
    c10::InefficientStdFunctionContext::makeDataPtr(src.storage().mutable_data_ptr().get(),
                                                    [](void*){}, at::kCPU);

  c10::Storage storage{c10::Storage::use_byte_size_t{}, size_bytes, std::move(data_ptr),
    /*allocator=*/&global_custom_alloc, /*resizeable=*/false};

  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  at::Tensor tensor = at::detail::make_tensor<c10::TensorImpl>(
       std::move(storage), cpu_ks, src.dtype());

  c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  tensor_impl->set_sizes_and_strides(sizes_, strides_);
  tensor_impl->set_storage_offset(storage_offset_);
  return tensor;
}

// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(
      self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1,
      "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(
      dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1,
      "Dummy test only allows copy from cpu -> dummy device.");

  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());

  if (self.is_contiguous() && dst.is_contiguous()) {
    std::memcpy(dst.storage().data_ptr().get(),
                self.storage().data_ptr().get(),
                self.storage().nbytes());
  } else {
    // Using cpu tensor to accomplishment stride copy.
    at::Tensor cpu_self = unsafe_create_cpu_tensor_from_dummy_tensor(self);
    at::Tensor cpu_dst = unsafe_create_cpu_tensor_from_dummy_tensor(dst);
    cpu_dst.copy_(cpu_self);
  }

  return dst;
}

at::Tensor custom__copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  return custom__copy_from(self, dst, false);
}

at::Tensor& custom_abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::abs_out(self, out);
}

at::Tensor custom_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  op_counter += 1;

  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return  at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, dtype);
}

at::Tensor custom_empty(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt, c10::optional<c10::MemoryFormat> optional_memory_format) {
  op_counter += 1;

  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return  at::detail::empty_generic(size, &global_custom_alloc, private_use_ks, dtype, optional_memory_format);
}

// This macro does the heavy lifting.
// With TORCH_LIBRARY_IMPL, you can register custom kernels for your backend.
// For open registration, we're registering all of our kernels to the PrivateUse1 dispatch key.
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("to.Device", &custom_to_device);
  m.impl("fill_.Scalar", &custom_fill__scalar);
  m.impl("_copy_from", &custom__copy_from);
  m.impl("_copy_from_and_resize", &custom__copy_from_and_resize);
  m.impl("empty_strided", &custom_empty_strided);
  m.impl("empty.memory_format", &custom_empty);
}

TORCH_LIBRARY_FRAGMENT(aten, m) {
  m.def(
    "_reinterpret_tensor(Tensor self, int[] size, int[] stride, int offset_increment=0) -> Tensor",
    torch::dispatch(
        c10::DispatchKey::AutogradPrivateUse1, _reinterpret_tensor),
    {at::Tag::pt2_compliant_tag});
}

void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("add.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("abs.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("sub.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("mul.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("div.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("pow.Tensor_Scalar", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("zero_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_foreach_add.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("index.Tensor", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("triu_indices", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("neg.out", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

bool custom_op_called() {
  bool called = false;
  if (op_counter > last_saved_value) {
    called = true;
    last_saved_value = op_counter;
  }
  return called;
}

class PrivateGeneratorImpl : public at::CPUGeneratorImpl {
public:
  // Constructors
  PrivateGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~PrivateGeneratorImpl() override = default;
};

// this is used to register generator
at::Generator make_generator_privateuse1(c10::DeviceIndex device_index) {
  return at::make_generator<PrivateGeneratorImpl>(device_index);
}

void register_generator() {
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
    m.def("custom_op_called", &custom_op_called, "check if our custom function was called");
    m.def("register_generator", &register_generator, "register generator for custom device");
}