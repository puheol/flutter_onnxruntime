# ONNX Runtime C++ API Guide

This document provides guidance on working with the ONNX Runtime C++ API in the Flutter ONNX Runtime plugin.

## API Overview

ONNX Runtime provides a C++ API that wraps the C API, making it more convenient to use in C++ code. The main classes you'll work with include:

- `Ort::Env`: The ONNX Runtime environment
- `Ort::Session`: An ONNX model session
- `Ort::MemoryInfo`: Information about memory allocations
- `Ort::Value`: Tensor values (inputs and outputs)
- `Ort::AllocatorWithDefaultOptions`: For memory allocation

## Important API Changes

The ONNX Runtime API has undergone changes in recent versions. Here are some key points to be aware of:

### String Allocations (v1.14+)

Recent versions of ONNX Runtime have changed how strings are returned from the API:

```cpp
// Old style (pre-v1.14)
const char* name = session->GetInputName(i, allocator);
// name is owned by the allocator, must be freed manually

// New style (v1.14+)
auto name_ptr = session->GetInputNameAllocated(i, allocator);
// name_ptr is a unique_ptr that will free the memory automatically
const char* name = name_ptr.get();
```

The newer API returns a `std::unique_ptr` with a custom deleter, improving memory safety. Our implementation uses this newer style.

## Working with Tensors

### Creating Input Tensors

```cpp
// Create input tensor
std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
std::vector<int64_t> input_shape = {1, 4};


auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, input_data.data(), input_data.size(),
    input_shape.data(), input_shape.size());
```

### Getting Output Tensors

```cpp
// Run inference
std::vector<const char*> input_names = {"input"};
std::vector<const char*> output_names = {"output"};
std::vector<Ort::Value> outputs = session->Run(
    Ort::RunOptions{nullptr}, 
    input_names.data(), &input_tensor, 1, 
    output_names.data(), 1);

// Access output data
float* output_data = outputs[0].GetTensorMutableData<float>();
```

## Memory Management

ONNX Runtime uses the RAII (Resource Acquisition Is Initialization) pattern extensively. Most objects manage their own memory through destructors.

### Smart Pointers

For string handling, ONNX Runtime returns `std::unique_ptr` objects with custom deleters:

```cpp
auto name_ptr = session->GetInputNameAllocated(i, allocator);
// Use name_ptr.get() to access the actual string
// Memory is automatically freed when name_ptr goes out of scope
```

### Tips for Memory Management

1. Use C++ smart pointers when possible to manage memory automatically.
2. Be aware of ownership semantics in the API - some functions transfer ownership, others don't.
3. For custom allocations, make sure to free them properly to avoid memory leaks.

## Error Handling

ONNX Runtime throws C++ exceptions for errors:

```cpp
try {
  // ONNX Runtime operations
} catch (const Ort::Exception& e) {
  // Handle the exception
  g_warning("ONNX Runtime error: %s", e.what());
}
```

It's good practice to wrap ONNX Runtime operations in try-catch blocks to handle errors gracefully.

## Additional Resources

- [Official ONNX Runtime C++ API Documentation](https://onnxruntime.ai/docs/api/c-cpp)
- [ONNX Runtime GitHub Repository](https://github.com/microsoft/onnxruntime)
- [C++ API Headers](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_cxx_api.h) 