# flutter_onnxruntime

Native Wrapper Flutter Plugin for ONNX Runtime

## 🌟 Why This Project?

`flutter_onnxruntime` is a lightweight plugin that provides native wrappers for running ONNX Runtime across multiple platforms without relying on pre-built libraries.

      📦 No Pre-built Libraries
      Libraries are fetched directly from official repositories during installation, ensuring they are always up-to-date!

      🪶 Lightweight Bundle Size
      The native implementation keeps your app slim and efficient.

      🛡️ Memory Safety
      All memory management is handled in native code, reducing the risk of memory leaks.

      🔄 Easy Upgrades
      Stay current with the latest ONNX Runtime releases without the hassle of maintaining complex generated FFI wrappers.

## Getting Started

### Installation

Add the following dependency to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_onnxruntime: ^1.0.0
```

### Quick Start

Example of running an addition model:
```dart
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

// create inference session
final ort = OnnxRuntime();
final session = await ort.createSessionFromAsset('assets/models/addition_model.onnx');

// specify input with data and shape
final inputs = {
   'A': await OrtValue.fromList([1, 1, 1], [3]),
   'B': await OrtValue.fromList([2, 2, 2], [3])
}

// start the inference
final outputs = await session.run(inputs);

// print output data
print(await outputs['C']!.asList());
```

To get started with the Flutter ONNX Runtime plugin, see the [API Usage Guide](docs/api_usage.md).

## Examples

### [Simple Addition Model](example/)

A simple model with only one operator (Add) that takes two inputs and produces one output.

Run the example with:
```bash
cd example
flutter pub get
flutter run
```

### Image Classification Model

A more complex model that takes an image as input and classifies it into one of the predefined categories.

## Component Overview

| Component | Description |
|-----------|-------------|
| OnnxRuntime | Main entry point for creating sessions and configuring global options |
| OrtSession | Represents a loaded ML model for running inference |
| OrtValue | Represents tensor data for inputs and outputs |
| OrtSessionOptions | Configuration options for session creation |
| OrtRunOptions | Configuration options for inference execution |

## Implementation Status

| Feature | Android | iOS | Linux | macOS | Windows | Web |
|---------|:-------:|:---:|:-----:|:-----:|:-------:|:---: |
| CPU Inference | ✅ | ✅ | ✅ | ✅ |   |   |
| Inference on Emulator | ✅ | ✅ | ✅ | ✅ |   |   |
| GPU Inference | ✅ | ✅ | ✅ | ✅ |   |   |
| Input/Output names | ✅ | ✅ | ✅ | ✅ |   |   |
| Input/Output Info | ✅ | ❌* | ✅ | ❌* |   |   |
| Model Metadata | ✅ | ❌* | ✅ | ❌* |   |   |
| Data Type Conversion | ✅ | ✅ | ✅ | ✅ |   |   |
| FP16 Support | ✅ | ❌** | ✍️ | ❌** |   |   |

✅: Complete

❌: Not supported

🚧: Ongoing

✍️: Planned

`*`: Retrieving model metadata and input/output info is not avialable in `onnxruntime-objc`, only names available.

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
