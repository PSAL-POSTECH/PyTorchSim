#pragma once

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>

#include <include/Macros.h>

namespace c10::openreg {

OPENREG_EXPORT bool is_amp_enabled();
OPENREG_EXPORT void set_amp_enabled(bool flag);
OPENREG_EXPORT at::ScalarType get_amp_dtype();
OPENREG_EXPORT void set_amp_dtype(at::ScalarType dtype);

} // namespace c10::openreg
