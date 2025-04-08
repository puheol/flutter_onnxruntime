
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

#### Testing

For testing, we use the `scripts/run_tests.sh` script to run unit and integration tests on all available platforms.

```
./scripts/run_tests.sh
```

You can also manually run tests for a specific platform:

1. Run unit tests:
    ```
    flutter test test/unit
    ```
2. Run integration tests:
    ```
    cd example
    flutter test integration_test/onnxruntime_integration_test.dart -d <device_id>
    ```
    or running via flutter drive:
    ```
    cd example
    flutter drive --driver=test_driver/integration_test.dart --target=integration_test/onnxruntime_integration_test.dart -d <device_id>
    ```
