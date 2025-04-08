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

| Feature | Status | Notes |
|---------|--------|-------|
| CPU Inference | ✅ Complete | Supported on all platforms |
| GPU Inference | ⚠️ Partial | Currently limited to specific platforms |
| Data Type Conversion | ✅ Complete | All major numeric types supported |
| Memory Management | ✅ Complete | Automatic and manual cleanup options |
| Model Metadata | ✅ Complete | Full access to model information (not available on iOS and macOS) |
| FP16 support | 🚧 Ongoing | In active development |
| Tensor manipulation | ❌ Planned | Scheduled for future release |


## Troubleshooting

### iOS
* Target minimum version: iOS 16
* "The 'Pods-Runner' target has transitive dependencies that include statically linked binaries: (onnxruntime-objc and onnxruntime-c)". In `Podfile` change:
    ```
    target 'Runner' do
    use_frameworks! :linkage => :static
    ```

### MacOS
* Target minimum version: MacOS 14
* "The 'Pods-Runner' target has transitive dependencies that include statically linked binaries: (onnxruntime-objc and onnxruntime-c)". In `Podfile` change:
    ```
    target 'Runner' do
    use_frameworks! :linkage => :static
    ```
* "error: compiling for macOS 10.14, but module 'flutter_onnxruntime' has a minimum deployment target of macOS 14.0". In terminal, cd to the `macos` directory and run the XCode to open the project:
    ```
    open Runner.xcworkspace
    ```
    Then change the "Minimum Deployments" to 14.0.

### Linux
* When running with ONNX Runtime 1.21.0, you may see reference counting warnings related to FlValue objects. These don't prevent the app from running but may be addressed in future updates.

## Contributing
Contributions to the Flutter ONNX Runtime plugin are welcome. Please see the [contributing.md](docs/contributing.md) file for more information.

#### Documentation
* For detailed Linux setup and troubleshooting:
   - [Linux Development Setup Guide](docs/linux/LINUX_SETUP.md)
   - [ONNX Runtime C++ API Guide](docs/linux/ONNX_RUNTIME_API.md)
   - [Flutter Linux Plugin Architecture](docs/linux/FLUTTER_LINUX_PLUGINS.md)

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and test on multiple platforms if possible
4. Submit a pull request with a clear description of your changes
