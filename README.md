# flutter_onnxruntime

Native Wrapper Flutter Plugin for ONNX Runtime

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
