#include <ATen/Context.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Dtype.h>

#include <runtime/OpenRegFunctions.h>
#include <amp/OpenRegAmp.h>
#include <include/openreg.h>
#include <functional>
#include <memory>
#include <thread>

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _getDefaultGenerator(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_default_generator expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  return THPGenerator_initDefaultGenerator(
      at::globalContext().defaultGenerator(
          c10::Device(c10::DeviceType::PrivateUse1, idx)));

  END_HANDLE_TH_ERRORS
}

PyObject* _setDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackLong(arg);

  torch::utils::device_lazy_init(at::kPrivateUse1);
  c10::openreg::set_device(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = c10::openreg::ExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDevice(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto device = static_cast<int32_t>(c10::openreg::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDeviceCount(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(c10::openreg::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* _isAutocastEnabled(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (c10::openreg::is_amp_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* _setAutocastEnabled(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyBool_Check(arg),
      "set_autocast_enabled expects a bool, but got ",
      THPUtils_typename(arg));
  c10::openreg::set_amp_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _getAutocastDtype(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPDtype* dtype_obj = torch::getTHPDtype(c10::openreg::get_amp_dtype());
  Py_INCREF(dtype_obj);
  return reinterpret_cast<PyObject*>(dtype_obj);
  END_HANDLE_TH_ERRORS
}

PyObject* _setAutocastDtype(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPDtype_Check(arg),
      "set_autocast_dtype expects a dtype, but got ",
      THPUtils_typename(arg));
  THPDtype* dtype_obj = reinterpret_cast<THPDtype*>(arg);
  at::ScalarType dtype = dtype_obj->scalar_type;
  c10::openreg::set_amp_dtype(dtype);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _getAmpSupportedDtype(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  PyObject* torch_mod = PyImport_ImportModule("torch");
  TORCH_CHECK(torch_mod != nullptr, "Failed to import torch module");

  PyObject* float16 = PyObject_GetAttrString(torch_mod, "float16");
  PyObject* float32 = PyObject_GetAttrString(torch_mod, "float32");

  PyObject* lst = PyList_New(1);
  PyList_SetItem(lst, 0, float32);
  //PyList_SetItem(lst, 1, float32);

  Py_DECREF(torch_mod);
  return lst;
  END_HANDLE_TH_ERRORS
}

// Stream functions
PyObject* _streamCreate(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  orStream_t stream = nullptr;
  orError_t err = orStreamCreate(&stream);
  std::cerr << "[DEBUG] Stream created: " << stream << std::endl;
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to create stream");
  }
  return THPUtils_packInt64(reinterpret_cast<int64_t>(stream));
  END_HANDLE_TH_ERRORS
}

PyObject* _streamCreateWithPriority(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_Size(args) == 2, "stream_create_with_priority expects 2 arguments");
  PyObject* flags_obj = PyTuple_GetItem(args, 0);
  PyObject* priority_obj = PyTuple_GetItem(args, 1);
  TORCH_CHECK(THPUtils_checkLong(flags_obj), "flags must be an int");
  TORCH_CHECK(THPUtils_checkLong(priority_obj), "priority must be an int");
  unsigned int flags = static_cast<unsigned int>(THPUtils_unpackLong(flags_obj));
  int priority = static_cast<int>(THPUtils_unpackLong(priority_obj));

  torch::utils::device_lazy_init(at::kPrivateUse1);
  orStream_t stream = nullptr;
  orError_t err = orStreamCreateWithPriority(&stream, flags, priority);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to create stream with priority");
  }
  return THPUtils_packInt64(reinterpret_cast<int64_t>(stream));
  END_HANDLE_TH_ERRORS
}

PyObject* _streamDestroy(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "stream_destroy expects an int");
  orStream_t stream = reinterpret_cast<orStream_t>(THPUtils_unpackLong(arg));
  orError_t err = orStreamDestroy(stream);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to destroy stream");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _streamSynchronize(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "stream_synchronize expects an int");
  orStream_t stream = reinterpret_cast<orStream_t>(THPUtils_unpackLong(arg));

  orError_t err;
  Py_BEGIN_ALLOW_THREADS
  err = orStreamSynchronize(stream);
  Py_END_ALLOW_THREADS

  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to synchronize stream");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _streamQuery(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "stream_query expects an int");
  orStream_t stream = reinterpret_cast<orStream_t>(THPUtils_unpackLong(arg));
  orError_t err = orStreamQuery(stream);
  if (err == orSuccess) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* _streamGetPriority(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "stream_get_priority expects an int");
  orStream_t stream = reinterpret_cast<orStream_t>(THPUtils_unpackLong(arg));
  int priority = 0;
  orError_t err = orStreamGetPriority(stream, &priority);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to get stream priority");
  }
  return THPUtils_packInt32(priority);
  END_HANDLE_TH_ERRORS
}

PyObject* _streamWaitEvent(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_Size(args) == 2, "stream_wait_event expects 2 arguments");
  PyObject* stream_obj = PyTuple_GetItem(args, 0);
  PyObject* event_obj = PyTuple_GetItem(args, 1);
  TORCH_CHECK(THPUtils_checkLong(stream_obj), "stream must be an int");
  TORCH_CHECK(THPUtils_checkLong(event_obj), "event must be an int");
  orStream_t stream = reinterpret_cast<orStream_t>(THPUtils_unpackLong(stream_obj));
  orEvent_t event = reinterpret_cast<orEvent_t>(THPUtils_unpackLong(event_obj));
  orError_t err = orStreamWaitEvent(stream, event, 0);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to wait for event");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Event functions
PyObject* _eventCreate(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  orEvent_t event = nullptr;
  orError_t err = orEventCreate(&event);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to create event");
  }
  return THPUtils_packInt64(reinterpret_cast<int64_t>(event));
  END_HANDLE_TH_ERRORS
}

PyObject* _eventCreateWithFlags(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "event_create_with_flags expects an int");
  unsigned int flags = static_cast<unsigned int>(THPUtils_unpackLong(arg));

  torch::utils::device_lazy_init(at::kPrivateUse1);
  orEvent_t event = nullptr;
  orError_t err = orEventCreateWithFlags(&event, flags);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to create event with flags");
  }
  return THPUtils_packInt64(reinterpret_cast<int64_t>(event));
  END_HANDLE_TH_ERRORS
}

PyObject* _eventDestroy(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "event_destroy expects an int");
  orEvent_t event = reinterpret_cast<orEvent_t>(THPUtils_unpackLong(arg));
  orError_t err = orEventDestroy(event);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to destroy event");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _eventRecord(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_Size(args) == 2, "event_record expects 2 arguments");
  PyObject* event_obj = PyTuple_GetItem(args, 0);
  PyObject* stream_obj = PyTuple_GetItem(args, 1);
  TORCH_CHECK(THPUtils_checkLong(event_obj), "event must be an int");
  TORCH_CHECK(THPUtils_checkLong(stream_obj), "stream must be an int");
  orEvent_t event = reinterpret_cast<orEvent_t>(THPUtils_unpackLong(event_obj));
  orStream_t stream = reinterpret_cast<orStream_t>(THPUtils_unpackLong(stream_obj));
  orError_t err = orEventRecord(event, stream);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to record event");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _eventSynchronize(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "event_synchronize expects an int");
  orEvent_t event = reinterpret_cast<orEvent_t>(THPUtils_unpackLong(arg));

  orError_t err;
  Py_BEGIN_ALLOW_THREADS
  err = orEventSynchronize(event);
  Py_END_ALLOW_THREADS

  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to synchronize event");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _eventQuery(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "event_query expects an int");
  orEvent_t event = reinterpret_cast<orEvent_t>(THPUtils_unpackLong(arg));
  orError_t err = orEventQuery(event);
  if (err == orSuccess) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* _eventElapsedTime(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_Size(args) == 2, "event_elapsed_time expects 2 arguments");
  PyObject* start_obj = PyTuple_GetItem(args, 0);
  PyObject* end_obj = PyTuple_GetItem(args, 1);
  TORCH_CHECK(THPUtils_checkLong(start_obj), "start event must be an int");
  TORCH_CHECK(THPUtils_checkLong(end_obj), "end event must be an int");
  orEvent_t start = reinterpret_cast<orEvent_t>(THPUtils_unpackLong(start_obj));
  orEvent_t end = reinterpret_cast<orEvent_t>(THPUtils_unpackLong(end_obj));
  float ms = 0.0f;
  orError_t err = orEventElapsedTime(&ms, start, end);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to get elapsed time");
  }
  return PyFloat_FromDouble(static_cast<double>(ms));
  END_HANDLE_TH_ERRORS
}

PyObject* _deviceSynchronize(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);

  orError_t err;
  Py_BEGIN_ALLOW_THREADS
  err = orDeviceSynchronize();
  Py_END_ALLOW_THREADS

  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to synchronize device");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _addTaskToStream(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_Size(args) == 2, "add_task_to_stream expects 2 arguments");
  PyObject* stream_obj = PyTuple_GetItem(args, 0);
  PyObject* callable_obj = PyTuple_GetItem(args, 1);

  TORCH_CHECK(THPUtils_checkLong(stream_obj), "stream must be an int");
  TORCH_CHECK(PyCallable_Check(callable_obj), "task must be callable");

  orStream_t stream = reinterpret_cast<orStream_t>(THPUtils_unpackLong(stream_obj));

  Py_INCREF(callable_obj);
  auto py_callable = std::shared_ptr<PyObject>(callable_obj, [](PyObject* obj) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    Py_DECREF(obj);
    PyGILState_Release(gstate);
  });

  auto task = [py_callable]() {
    PyGILState_STATE gstate = PyGILState_Ensure();
    try {
      PyObject* result = PyObject_CallObject(py_callable.get(), nullptr);
      if (result == nullptr) {
        PyErr_Print();
        PyErr_Clear();
      } else {
        Py_DECREF(result);
      }
    } catch (...) {
    }

    PyGILState_Release(gstate);
  };
  orError_t err = openreg::addTaskToStream(stream, task);
  if (err != orSuccess) {
    TORCH_CHECK(false, "Failed to add task to stream");
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {"_get_device", _getDevice, METH_NOARGS, nullptr},
    {"_set_device", _setDevice, METH_O, nullptr},
    {"_exchangeDevice", _exchangeDevice, METH_O, nullptr},
    {"_get_device_count", _getDeviceCount, METH_NOARGS, nullptr},
    {"is_autocast_enabled", _isAutocastEnabled, METH_NOARGS, nullptr},
    {"set_autocast_enabled", _setAutocastEnabled, METH_O, nullptr},
    {"get_autocast_dtype", _getAutocastDtype, METH_NOARGS, nullptr},
    {"set_autocast_dtype", _setAutocastDtype, METH_O, nullptr},
    {"get_amp_supported_dtype", _getAmpSupportedDtype, METH_NOARGS, nullptr},
    // Stream functions
    {"_stream_create", _streamCreate, METH_NOARGS, nullptr},
    {"_stream_create_with_priority", _streamCreateWithPriority, METH_VARARGS, nullptr},
    {"_stream_destroy", _streamDestroy, METH_O, nullptr},
    {"_stream_synchronize", _streamSynchronize, METH_O, nullptr},
    {"_stream_query", _streamQuery, METH_O, nullptr},
    {"_stream_get_priority", _streamGetPriority, METH_O, nullptr},
    {"_stream_wait_event", _streamWaitEvent, METH_VARARGS, nullptr},
    // Event functions
    {"_event_create", _eventCreate, METH_NOARGS, nullptr},
    {"_event_create_with_flags", _eventCreateWithFlags, METH_O, nullptr},
    {"_event_destroy", _eventDestroy, METH_O, nullptr},
    {"_event_record", _eventRecord, METH_VARARGS, nullptr},
    {"_event_synchronize", _eventSynchronize, METH_O, nullptr},
    {"_event_query", _eventQuery, METH_O, nullptr},
    {"_event_elapsed_time", _eventElapsedTime, METH_VARARGS, nullptr},
    // Device functions
    {"_device_synchronize", _deviceSynchronize, METH_NOARGS, nullptr},
    // Stream task functions
    {"_add_task_to_stream", _addTaskToStream, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

/*
 * When ASAN is enabled, PyTorch modifies the dlopen flag during import,
 * causing all global and weak symbols in _C.so and its dependent libraries
 * to be exposed to the global symbol scope, which in turn causes
 * subsequent symbols with the same name in other libraries to be intercepted.
 * Therefore, it cannot be named initModule here, otherwise initModule
 * in torch/csrc/Module.cpp will be called, resulting in failure.
 */
extern "C" OPENREG_EXPORT PyObject* initOpenRegModule(void) {
  static struct PyModuleDef openreg_C_module = {
      PyModuleDef_HEAD_INIT, "torch_openreg._C", nullptr, -1, methods};
  PyObject* mod = PyModule_Create(&openreg_C_module);

  return mod;
}
