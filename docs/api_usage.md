# Flutter ONNX Runtime API Usage Guide

This guide provides examples of how to use the Flutter ONNX Runtime plugin to run machine learning models in your Flutter applications.

## Installation

Add the following dependency to your `pubspec.yaml`:

```yaml
dependencies:
  flutter_onnxruntime: ^1.0.0
```

## Basic Usage

### Importing the Library

```dart
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
```

### Initializing

The ONNX Runtime is automatically initialized when you create your first session.

```dart
// Optional: Check platform version
final platformVersion = await OnnxRuntime.platformVersion;
print('Running on $platformVersion');
```

### Creating a Session

```dart
// Create a session from a model file
final session = await OnnxRuntime.createSession(
  modelPath: 'assets/model.onnx',
  // Optional session configuration
  sessionOptions: OrtSessionOptions(
    intraOpNumThreads: 2,
    interOpNumThreads: 1,
    providers: ['CPU'],
    useArena: true,
    deviceId: 0,
  ),
);

// Session information
print('Input names: ${session.inputNames}');
print('Output names: ${session.outputNames}');
```

* Optional: to get available providers

  ```dart
  final providers = await OnnxRuntime.getAvailableProviders();
  print('Available providers: $providers');
  ```

### Running Inference

```dart
// Prepare input data - assuming a model with one input named 'input'
// with shape [1, 3, 224, 224] (batch_size, channels, height, width)
final inputData = Float32List(1 * 3 * 224 * 224);
// ... fill inputData with your values ...

// Create inputs map
final inputs = {
  'input': inputData,
  'input_shape': [1, 3, 224, 224], // Specify shape
};

// Run inference
final outputs = await session.run(inputs);

// Process results - outputs will be a map with keys for each output tensor
final outputTensor = outputs['output'];
final outputShape = outputs['output_shape'];
print('Output shape: $outputShape');
```

### Closing the Session

```dart
// Always close the session when done to free resources
await session.close();
```

## Working with OrtValue

The `OrtValue` class provides a more flexible way to manage tensors:

### Creating Tensors

```dart
// Create from Float32List
final inputTensor = await OrtValue.fromList(
  Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
  [2, 2], // Shape: 2x2 matrix
);

// Create from Int32List
final intTensor = await OrtValue.fromList(
  Int32List.fromList([1, 2, 3, 4]),
  [4], // Shape: vector of 4 elements
);

// Create from Uint8List (for images)
final imageTensor = await OrtValue.fromList(
  imageBytes, // Uint8List from an image
  [1, height, width, channels],
);
```

### Data Type Conversion

```dart
// Convert to different data type
final float16Tensor = await inputTensor.to(OrtDataType.float16);
```

### Accessing Tensor Data

```dart
// Get data as Float32List
final floatData = await inputTensor.asFloat32List();

// Get data as Int32List
final intData = await intTensor.asInt32List();
```

### Using OrtValue with Session Run

```dart
// Create OrtValue tensor
final inputTensor = await OrtValue.fromList(
  Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
  [2, 2],  // Shape: 2x2 matrix
);

// Create another tensor for a second input
final inputTensor2 = await OrtValue.fromList(
  Float32List.fromList([5.0, 6.0, 7.0, 8.0]),
  [2, 2],
);

// Use OrtValue objects directly with session.run()
final inputs = {
  'input1': inputTensor,
  'input2': inputTensor2,
};

// Run inference with OrtValue inputs
final outputs = await session.run(inputs);

// Process results - outputs will be a map with keys for each output tensor
final outputTensor = outputs['output'];
final outputShape = outputs['output_shape'];

// Clean up resources
await inputTensor.dispose();
await inputTensor2.dispose();
```

### Memory Management

```dart
// Explicitly release native resources when done
await inputTensor.dispose();
await float16Tensor.dispose();
```

## Advanced Usage

### Getting Model Metadata

```dart
// Get model metadata
final metadata = await session.getMetadata();
print('Producer: ${metadata.producerName}');
print('Graph name: ${metadata.graphName}');
print('Domain: ${metadata.domain}');
print('Description: ${metadata.description}');
print('Version: ${metadata.version}');
```

### Getting Input/Output Information

```dart
// Get detailed input information
final inputInfo = await session.getInputInfo();
for (final info in inputInfo) {
  print('Input: ${info['name']}');
  print('  Shape: ${info['shape']}');
  print('  Type: ${info['type']}');
}

// Get detailed output information
final outputInfo = await session.getOutputInfo();
for (final info in outputInfo) {
  print('Output: ${info['name']}');
  print('  Shape: ${info['shape']}');
  print('  Type: ${info['type']}');
}
```

### Custom Run Options

```dart
// Configure runtime options for inference
final results = await session.run(
  inputs,
  options: OrtRunOptions(
    logSeverityLevel: 2,
    logVerbosityLevel: 4,
    terminate: false,
  ),
);
```

## Best Practices

1. **Resource Management**
   - Always call `session.close()` and `tensor.dispose()` when done
   - Use try/finally blocks to ensure resources are released

2. **Performance Optimization**
   - Reuse OrtValue instances when possible
   - Prefer operating on batches of data rather than individual items
   - Use appropriate data types (e.g., Float32List is generally more efficient than List<double>)

3. **Cross-Platform Development**
   - Test on all target platforms, as performance characteristics may vary
   - Be aware that some devices may not support all features (e.g., GPU acceleration)

4. **Memory Efficiency**
   - Dispose of large tensors immediately after use
   - Be mindful of tensor shapes and sizes, especially for mobile devices 