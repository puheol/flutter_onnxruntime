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
    Map<String, dynamic> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    // Store the inputs for later assertions
    lastRunInputs = inputs;

    // Return mock outputs
    return Future.value({
      'outputs': {
        'output1': [1.0, 2.0, 3.0, 4.0],
        'output1_shape': [2, 2],
      },
    });
  }

  @override
  Future<Map<String, dynamic>> createOrtValue(
    String sourceType,
    dynamic data,
    List<int> shape,
    String targetType,
    String device,
  ) {
    // Return a mock OrtValue map
    return Future.value({
      'valueId': 'test_value_id_${DateTime.now().millisecondsSinceEpoch}',
      'dataType': targetType,
      'shape': shape,
      'device': device,
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
  Future<Map<String, dynamic>> getOrtValueData(String valueId, String dataType) {
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
    test('run() should accept OrtValue objects as inputs', () async {
      // Create OrtValue tensors
      final tensor1 = await OrtValue.fromFloat32List(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      final tensor2 = await OrtValue.fromFloat32List(Float32List.fromList([5.0, 6.0, 7.0, 8.0]), [2, 2]);

      // Use OrtValue objects in inputs map
      final inputs = {'input1': tensor1, 'input2': tensor2};

      // Run inference with OrtValues
      await session.run(inputs);

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
    });

    test('run() should accept mixed inputs (OrtValue and raw data)', () async {
      // Create one OrtValue tensor
      final tensor = await OrtValue.fromFloat32List(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      // Create inputs with both OrtValue and raw data
      final inputs = {
        'input1': tensor,
        'input2': [5.0, 6.0, 7.0, 8.0],
        'input2_shape': [2, 2],
      };

      // Run inference with mixed inputs
      await session.run(inputs);

      // Verify the inputs were correctly processed
      expect(mockPlatform.lastRunInputs, isNotNull);

      // Check that the OrtValue was processed correctly
      expect(mockPlatform.lastRunInputs!['input1'], isA<Map<String, dynamic>>());
      expect(mockPlatform.lastRunInputs!['input1']['valueId'], tensor.id);
      expect(mockPlatform.lastRunInputs!['input1_shape'], tensor.shape);

      // Check that the raw data was passed through unchanged
      expect(mockPlatform.lastRunInputs!['input2'], [5.0, 6.0, 7.0, 8.0]);
      expect(mockPlatform.lastRunInputs!['input2_shape'], [2, 2]);
    });

    test('should properly clean up OrtValue resources', () async {
      // This test verifies that dispose() can be called after using in session.run()

      // Create an OrtValue tensor
      final tensor = await OrtValue.fromFloat32List(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      // Use OrtValue in session.run()
      await session.run({'input1': tensor});

      // Should be able to dispose without errors
      await expectLater(tensor.dispose(), completes);
    });
  });
}
