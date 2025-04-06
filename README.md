# flutter_onnxruntime

Native Wrapper Flutter Plugin for ONNX Runtime

## 🌟 Why This Project?

      📦 No Pre-built Libraries Needed
         Libraries are pulled directly from official repositories during installation - always up-to-date!
   
      🪶 Lightweight Bundle Size
         Native implementation keeps your app slim and efficient without bloated dependencies.
   
      🛡️ Memory Safety First
         Reduced risk of memory leaks with native-side memory management rather than handling in Dart.
   
      🔄 Future-Proof Adaptability
         Easily adapt to new OnnxRuntime releases without maintaining complex generated FFI wrappers.

## Getting Started

## Know issues

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
* The plugin now supports Linux platforms through a dedicated C++ implementation.
* When running with ONNX Runtime 1.21.0, you may see reference counting warnings related to FlValue objects. These don't prevent the app from running but may be addressed in future updates.
* For Linux-specific setup and troubleshooting, see our [detailed Linux setup guide](docs/linux/LINUX_SETUP.md).

## Contributing

We welcome contributions to improve the flutter_onnxruntime plugin! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### Setting Up Development Environment

#### Pre-commit Setup
We use a pre-commit hook to ensure code quality and consistency. Follow these steps to set it up:

1. Install required tools:
   - Dart SDK and Flutter (required for all platforms)
   - **ktlint** (for Android Kotlin formatting):
     ```
     curl -sSLO https://github.com/pinterest/ktlint/releases/download/1.0.0/ktlint && chmod a+x ktlint && sudo mv ktlint /usr/local/bin/
     ```
   - **SwiftLint** (for iOS/macOS Swift formatting, macOS only):
     ```
     brew install swiftlint
     ```
   - **clang-format** (for C++ formatting):
     ```
     # Ubuntu/Debian
     sudo apt-get install clang-format
     
     # macOS
     brew install clang-format
     ```

2. Copy the pre-commit hook to your local Git hooks directory:
   ```
   cp hooks/pre-commit .git/hooks/
   chmod +x .git/hooks/pre-commit
   ```

The pre-commit hook will:
- Format Dart code
- Format Kotlin code (Android)
- Format Swift code (iOS/macOS)
- Format C++ code
- Run Flutter analyze
- Prevent commits with formatting errors

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
