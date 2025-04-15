# Flutter ONNX Runtime Plugin - Linux Architecture

## Overview

This document outlines the architecture for the implementation of the Flutter ONNX Runtime plugin for Linux. The goal is to create a clean, maintainable, and robust implementation that follows C++ best practices while providing efficient and reliable access to the ONNX Runtime from Flutter applications.

## Design Goals

1. **Clear separation of concerns** - Modular components with well-defined responsibilities
2. **RAII-based resource management** - Using C++ idioms for safe memory management
3. **Minimal overhead** - Avoiding unnecessary serialization/deserialization steps
4. **Type-safety** - Leveraging C++ type system to prevent errors
5. **Testability** - Components designed to be easily testable in isolation
6. **Robust error handling** - Clear error reporting and propagation
7. **Thread safety** - Safe operation in multi-threaded Flutter applications

## High-Level Architecture

The plugin will consist of several primary components:

1. **Plugin Entry Point** (`FlutterOnnxruntimePlugin`) - Handles method channel calls and dispatches to appropriate managers
2. **Session Manager** - Handles creation, management, and cleanup of ONNX Runtime sessions
3. **Tensor Manager** - Manages OrtValue objects (tensors) with proper lifetime handling
4. **Value Conversion Utilities** - Converts between Flutter values and C++ types
5. **Run Options Provider** - Manages inference run options and configurations

```
┌───────────────────────────┐
│  Flutter (Dart)           │
└───────────┬───────────────┘
            │ Method Channel
┌───────────▼───────────────┐
│  FlutterOnnxruntimePlugin │
└───────────┬───────────────┘
            │
  ┌─────────┴──────────┐
  │                    │
┌─▼────────────┐  ┌────▼────────┐   ┌───────────────────┐
│SessionManager│  │TensorManager│───►ValueConversionUtil│
└──────┬───────┘  └─────┬───────┘   └───────────────────┘
       │                │               ▲
       │                │               │
       │         ┌──────▼────────┐      │
       └────────►│RunOptionsUtil │──────┘
                 └───────────────┘
                       │
┌──────────────────────▼─────────┐
│     ONNX Runtime C++ API       │
└────────────────────────────────┘
```

## Component Details

### 1. FlutterOnnxruntimePlugin

The main plugin class responsible for:
- Registering with the Flutter engine
- Receiving method calls from Dart
- Dispatching calls to the appropriate managers
- Converting return values to `FlValue` for Dart
- Error handling and reporting

Key methods:
- `handle_method_call` - Dispatcher for all incoming method calls
- `getPlatformVersion` - Returns Linux version
- Methods for each supported operation (createSession, runInference, etc.)

### 2. SessionManager

Manages ONNX Runtime sessions:
- Creates and stores `Ort::Session` objects
- Generates unique session IDs
- Provides sessions when needed for inference
- Handles session cleanup
- Manages session options (execution provider selection, graph optimization, etc.)

Key methods:
- `createSession` - Creates a new ONNX Runtime session
- `getSession` - Retrieves a session by ID
- `closeSession` - Closes and removes a session
- `getSessionInfo` - Retrieves metadata about a session
- `configureSessionOptions` - Applies configuration options to a session

### 3. TensorManager

Manages OrtValue (tensor) objects:
- Creates tensors from various data sources
- Stores tensors with unique IDs
- Handles tensor conversions between types
- Manages tensor lifecycle and memory
- Thread-safe operations on tensors

Key methods:
- `createTensor` - Creates a new tensor from Flutter data
- `getTensor` - Retrieves a tensor by ID
- `releaseTensor` - Frees tensor resources
- `getTensorData` - Extracts data from a tensor
- `convertTensor` - Converts a tensor to a different data type
### 4. ValueConversionUtil

Utility class for type conversion:
- Converts between `FlValue` and C++ types
- Handles tensor data extraction and conversion
- Manages shape information
- Provides efficient memory reuse when possible

Key methods:
- `flValueToVector<T>` - Converts Flutter lists to C++ vectors
- `vectorToFlValue<T>` - Converts C++ vectors to Flutter lists
- `flValueToTensorData` - Extracts tensor data from Flutter values
- `tensorDataToFlValue` - Converts tensor data to Flutter values

### 5. RunOptionsUtil

Utility for managing ONNX Runtime run options:
- Configures inference execution parameters
- Sets timeout and thread pool settings
- Manages execution providers
- Configures memory patterns and optimizations

Key methods:
- `createRunOptions` - Creates Ort::RunOptions from Flutter configuration
- `configureExecutionProviders` - Sets up execution providers (CPU/GPU)
- `configureThreading` - Manages thread pool settings for inference

## Memory Management

The new implementation will leverage C++ RAII principles:
- `std::unique_ptr` for managing `Ort::Session` and `Ort::Value` objects
- `std::vector` for tensor data storage
- Clear ownership rules for all resources
- No raw `new/delete` or `malloc/free` usage
- Exception safety throughout the codebase
- Memory reuse for frequently allocated buffers where appropriate

## Implementation Details

### Tensor Creation and Management

1. When a tensor is created from Dart data:
   - Extract type, shape, and data from `FlValue` parameters
   - Create appropriate C++ vector to hold data
   - Create an `Ort::Value` using the data
   - Store both the `Ort::Value` and the data vector in a wrapper class
   - Generate and return a unique ID

2. For inference:
   - Retrieve tensors by ID
   - Run inference using direct `Ort::Value` pointers
   - Create result tensors and store them
   - Return IDs of result tensors to Dart

3. Memory approach:
   - Use `Ort::MemoryInfo` appropriately to control tensor memory allocation
   - Leverage ONNX Runtime's memory management when possible
   - Implement a simple buffer reuse strategy for frequent tensor operations

### Session Management

1. Session creation:
   - Support both model path and model buffer loading
   - Apply session options from Dart configuration
   - Configure appropriate execution providers
   - Support custom execution providers when available

2. Session options:
   - Thread pool configuration
   - Execution mode (sequential/parallel)
   - Graph optimization level
   - Memory pattern optimization
   - Execution provider selection and configuration

### Thread Safety

The implementation will ensure:
- Thread-safe access to session and tensor managers using mutexes
- Safe concurrent model inference operations
- Protection against race conditions when sharing tensors
- Thread-local temporary allocations where appropriate

### Error Handling

Use C++ exceptions internally, catching and converting to Flutter error responses at the method channel boundary:
- Custom exception types for different error categories
- Detailed error messages including context
- Consistent error reporting format
- Proper stack unwinding and resource cleanup during errors

## Directory Structure

```
linux/
├── CMakeLists.txt                       # Build configuration
├── include/
│   └── flutter_onnxruntime/
│       ├── flutter_onnxruntime_plugin.h # Public plugin interface
│       └── export.h                     # Export macros
├── src/
│   ├── flutter_onnxruntime_plugin.cc    # Plugin implementation
│   ├── session_manager.h                # Session manager header
│   ├── session_manager.cc               # Session manager implementation
│   ├── tensor_manager.h                 # Tensor manager header
│   ├── tensor_manager.cc                # Tensor manager implementation
│   ├── value_conversion.h               # Value conversion utilities header
│   ├── value_conversion.cc              # Value conversion utilities implementation
│   ├── tensor_data.h                    # Tensor data wrapper class
│   └── exceptions.h                     # Custom exception classes
└── test/
    ├── flutter_onnxruntime_plugin_test.cc # Plugin tests
    ├── session_manager_test.cc            # Session manager tests
    ├── tensor_manager_test.cc             # Tensor manager tests
    ├── value_conversion_test.cc           # Value conversion tests
    └── run_options_test.cc                # Run options tests
```

## API Design

### Method Channel Interface

The plugin will expose the following methods:

1. `getPlatformVersion` - Returns the platform version
2. `createSession` - Creates a new ONNX Runtime session
3. `runInference` - Runs inference using a session
4. `closeSession` - Closes a session
5. `getMetadata` - Gets metadata about a session
6. `getInputInfo` - Gets information about session inputs
7. `getOutputInfo` - Gets information about session outputs
8. `createOrtValue` - Creates a tensor
9. `convertOrtValue` - Converts a tensor to a different type
10. `getOrtValueData` - Gets data from a tensor
11. `releaseOrtValue` - Releases a tensor
12. `getAvailableExecutionProviders` - Lists available execution providers
13. `getMemoryInfo` - Gets memory usage information
