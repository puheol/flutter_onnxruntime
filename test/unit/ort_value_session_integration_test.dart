import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterOnnxruntimePlatform with MockPlatformInterfaceMixin implements FlutterOnnxruntimePlatform {
  // Store values for assertions
  Map<String, dynamic>? lastRunInputs;

  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) {
    return Future.value({
      'sessionId': 'test_session_id',
      'inputNames': ['input1', 'input2'],
      'outputNames': ['output1'],
    });
  }

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, OrtValue> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    // Store the inputs for later assertions
    lastRunInputs = {
      for (final entry in inputs.entries) entry.key: {'valueId': entry.value.id},
    };

    // Add shape info for each input
    for (final entry in inputs.entries) {
      lastRunInputs!['${entry.key}_shape'] = entry.value.shape;
    }

    // Return mock outputs - now with OrtValue properties
    return Future.value({
      'output1': [
        'tensor_001',
        'float32',
        [2, 2],
      ],
    });
  }

  @override
  Future<Map<String, dynamic>> createOrtValue(String sourceType, dynamic data, List<int> shape) {
    // Return a mock OrtValue map
    return Future.value({
      'valueId': 'test_value_id_${DateTime.now().millisecondsSinceEpoch}',
      'dataType': sourceType,
      'shape': shape,
      'device': 'cpu',
    });
  }

  @override
  Future<void> closeSession(String sessionId) => Future.value();

  @override
  Future<Map<String, dynamic>> getMetadata(String sessionId) {
    return Future.value({
      'producerName': 'Test Producer',
      'graphName': 'Test Graph',
      'domain': 'test.domain',
      'description': 'Test Description',
      'version': 1,
      'customMetadataMap': {},
    });
  }

  @override
  Future<List<Map<String, dynamic>>> getInputInfo(String sessionId) {
    return Future.value([
      {
        'name': 'input1',
        'shape': [2, 2],
        'type': 'FLOAT',
      },
      {
        'name': 'input2',
        'shape': [2, 2],
        'type': 'FLOAT',
      },
    ]);
  }

  @override
  Future<List<Map<String, dynamic>>> getOutputInfo(String sessionId) {
    return Future.value([
      {
        'name': 'output1',
        'shape': [2, 2],
        'type': 'FLOAT',
      },
    ]);
  }

  @override
  Future<Map<String, dynamic>> convertOrtValue(String valueId, String targetType) {
    return Future.value({
      'valueId': valueId,
      'dataType': targetType,
      'shape': [2, 2],
      'device': 'cpu',
    });
  }

  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId) {
    return Future.value({
      'data': [1.0, 2.0, 3.0, 4.0],
      'shape': [2, 2],
    });
  }

  @override
  Future<Map<String, dynamic>> moveOrtValueToDevice(String valueId, String targetDevice) {
    return Future.value({
      'valueId': valueId,
      'dataType': 'float32',
      'shape': [2, 2],
      'device': targetDevice,
    });
  }

  @override
  Future<void> releaseOrtValue(String valueId) => Future.value();
}

class ConversionTrackingMock extends MockFlutterOnnxruntimePlatform {
  String? lastConvertedValueId;
  String? lastConvertedTargetType;

  @override
  Future<Map<String, dynamic>> convertOrtValue(String valueId, String targetType) {
    // Track the conversion operation for assertion
    lastConvertedValueId = valueId;
    lastConvertedTargetType = targetType;

    return Future.value({
      'valueId': 'converted_$valueId',
      'dataType': targetType,
      'shape': [2, 2],
      'device': 'cpu',
    });
  }
}

void main() {
  late OrtSession session;
  late MockFlutterOnnxruntimePlatform mockPlatform;
  final FlutterOnnxruntimePlatform initialPlatform = FlutterOnnxruntimePlatform.instance;

  setUp(() async {
    mockPlatform = MockFlutterOnnxruntimePlatform();
    FlutterOnnxruntimePlatform.instance = mockPlatform;

    // Create a session for testing
    final onnxRuntime = OnnxRuntime();
    session = await onnxRuntime.createSession('test_model.onnx');
  });

  tearDown(() {
    FlutterOnnxruntimePlatform.instance = initialPlatform;
  });

  group('OrtSession with OrtValue integration', () {
    test('run() should accept only OrtValue objects as inputs', () async {
      // Create OrtValue tensors
      final tensor1 = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);
      final tensor2 = await OrtValue.fromList(Float32List.fromList([5.0, 6.0, 7.0, 8.0]), [2, 2]);

      // Use OrtValue objects in inputs map
      final inputs = {'input1': tensor1, 'input2': tensor2};

      // Run inference with OrtValues
      final outputs = await session.run(inputs);

      // Verify that the inputs were correctly processed
      expect(mockPlatform.lastRunInputs, isNotNull);

      // Check that the valueId and shape were passed correctly for input1
      expect(mockPlatform.lastRunInputs!['input1'], isA<Map<String, dynamic>>());
      expect(mockPlatform.lastRunInputs!['input1']['valueId'], tensor1.id);
      expect(mockPlatform.lastRunInputs!['input1_shape'], tensor1.shape);

      // Check that the valueId and shape were passed correctly for input2
      expect(mockPlatform.lastRunInputs!['input2'], isA<Map<String, dynamic>>());
      expect(mockPlatform.lastRunInputs!['input2']['valueId'], tensor2.id);
      expect(mockPlatform.lastRunInputs!['input2_shape'], tensor2.shape);

      // Verify output format
      expect(outputs, isA<Map<String, OrtValue>>());
      expect(outputs.keys, contains('output1'));
      expect(outputs['output1']!.id, 'tensor_001');
      expect(outputs['output1']!.shape, [2, 2]);
    });

    test('should properly clean up OrtValue resources', () async {
      // This test verifies that dispose() can be called after using in session.run()

      // Create an OrtValue tensor
      final tensor = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      // Use OrtValue in session.run()
      await session.run({'input1': tensor});

      // Should be able to dispose without errors
      await expectLater(tensor.dispose(), completes);
    });

    test('run output should be OrtValue that can be used in subsequent operations', () async {
      // Use custom mock that tracks conversion operations
      final conversionMock = ConversionTrackingMock();
      FlutterOnnxruntimePlatform.instance = conversionMock;

      // Setup the createSession and runInference methods
      final onnxRuntime = OnnxRuntime();
      final testSession = await onnxRuntime.createSession('test_model.onnx');

      // Create input tensor
      final inputTensor = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      // Run inference
      final outputs = await testSession.run({'input1': inputTensor});

      // Get the output tensor
      final outputTensor = outputs['output1']!;

      // Use the output tensor in another operation (e.g., convert to int32)
      await outputTensor.to(OrtDataType.int32);

      // Verify that the conversion was called with the right parameters
      expect(conversionMock.lastConvertedValueId, 'tensor_001');
      expect(conversionMock.lastConvertedTargetType, 'int32');

      // Restore original mock
      FlutterOnnxruntimePlatform.instance = mockPlatform;
    });
  });
}
