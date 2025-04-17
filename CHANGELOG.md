## 1.1.0
* Standardize error codes and error messages from native
* Remove auto disposal of input tensors after inference in Kotlin
* Refactor C++ to return standard Platform Exception and impose standard error handling
* Add back the example to package as required by pub.dev; this will increase the package size unnecessarily

## 1.0.0
* Support running inference with an ONNX model on Android, iOS, Linux, and macOS
* Support ONNX Runtime version 1.21.0