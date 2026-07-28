#include <ATen/autocast_mode.h>
#include <iostream>
#include "OpenRegAmp.h"

namespace {
  bool g_amp_enabled = false;
  at::ScalarType g_amp_dtype = at::kFloat;
}

namespace c10::openreg {

OPENREG_EXPORT bool is_amp_enabled() {
  return g_amp_enabled;
}

OPENREG_EXPORT void set_amp_enabled(bool flag) {
  g_amp_enabled = flag;
}

OPENREG_EXPORT at::ScalarType get_amp_dtype() {
  return g_amp_dtype;
}

OPENREG_EXPORT void set_amp_dtype(at::ScalarType dtype) {
  g_amp_dtype = dtype;
}

} // namespace c10::openreg
