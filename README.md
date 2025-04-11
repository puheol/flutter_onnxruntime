# flutter_onnxruntime

Native Wrapper Flutter Plugin for ONNX Runtime

## 🌟 Why This Project?

This onnxruntime plugin uses native wrappers to run `onnxruntime` on different platforms instead of using `dart:ffi` to wrap pre-built `onnxruntime` libraries.

      📦 No Pre-built Libraries Needed
         Libraries are pulled directly from official repositories during installation - always up-to-date!
   
      🪶 Lightweight Bundle Size
         Native implementation keeps your app slim and efficient.
   
      🛡️ Memory Safety First
         Reduced risk of memory leaks with native-side memory management rather than handling in Dart.
   
      🔄 Future-Proof Adaptability
         Easily adapt to new OnnxRuntime releases without maintaining complex generated FFI wrappers.

## Getting Started

### Installation

Add the following dependency to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_onnxruntime: ^1.0.0
```

To get started with the Flutter ONNX Runtime plugin, see the [API Usage Guide](docs/api_usage.md).

## Component Overview

| Component | Description |
|-----------|-------------|
| OnnxRuntime | Main entry point for creating sessions and configuring global options |
| OrtSession | Represents a loaded ML model for running inference |
| OrtValue | Represents tensor data for inputs and outputs |
| OrtSessionOptions | Configuration options for session creation |
| OrtRunOptions | Configuration options for inference execution |

## Implementation Status

| Feature | Android | iOS | Linux | macOS | Windows |
|---------|:-------:|:---:|:-----:|:-----:|:-------:|
| CPU Inference | ✅ | ✅ | ✅ | ✅ | ✍️ |
| Inference on Emulator | ✅ | ✅ | ✅ | ✅ | ✍️ |
| GPU Inference | 🚧 | 🚧 | 🚧 | 🚧 | ✍️ |
| Input/Output names | ✅ | ✅ | ✅ | ✅ | ✍️ |
| Input/Output Info | ✅ | ❌* | ✅ | ❌* | ✍️ |
| Model Metadata | ✅ | ❌* | ✅ | ❌* | ✍️ |
| Tensor Management | ✅ | ✅ | ✅ | ✅ | ✍️ |
| Data Type Conversion | ✅ | ✅ | ✅ | ✅ | ✍️ |
| FP16 Support | ✅ | ❌** | 🚧 | ❌** | ✍️ |
| Tensor Manipulation | ✍️ | ✍️ | ✍️ | ✍️ | ✍️ |

✅: Complete

❌: Not supported

🚧: Ongoing

✍️: Planned

`*`: ORT does not support retrieving model metadata and input/output info, we can only retrieve the input/output names.

`**`: Swift does not support FP16 type.

## Troubleshooting

For troubleshooting, see the [troubleshooting.md](docs/troubleshooting.md) file.

## Contributing
Contributions to the Flutter ONNX Runtime plugin are welcome. Please see the [contributing.md](docs/contributing.md) file for more information.

#### Documentation
* For detailed Linux setup and troubleshooting:
   - [Linux Development Setup Guide](docs/linux/LINUX_SETUP.md)
   - [ONNX Runtime C++ API Guide](docs/linux/ONNX_RUNTIME_API.md)
   - [Flutter Linux Plugin Architecture](docs/linux/FLUTTER_LINUX_PLUGINS.md)
