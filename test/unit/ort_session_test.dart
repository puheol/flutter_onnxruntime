import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:flutter_onnxruntime/src/ort_model_metadata.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterOnnxruntimePlatform with MockPlatformInterfaceMixin implements FlutterOnnxruntimePlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) {
    return Future.value({
      'sessionId': 'test_session_id',
      'inputNames': ['input1', 'input2'],
      'outputNames': ['output1', 'output2'],
    });
  }

  // Track method calls for verification
  String? lastSessionIdForRun;
  Map<String, dynamic>? lastInputsForRun;
  Map<String, dynamic>? lastRunOptions;

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, dynamic> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    // Track the invocation for verification
    lastSessionIdForRun = sessionId;
    lastInputsForRun = inputs;
    lastRunOptions = runOptions;

    // Return mock output
    return Future.value({
      'output1': [1.0, 2.0, 3.0],
      'output2': [
        [4.0, 5.0],
        [6.0, 7.0],
      ],
    });
  }

  // Track session close calls
  String? lastClosedSessionId;

  @override
  Future<void> closeSession(String sessionId) {
    lastClosedSessionId = sessionId;
    return Future.value();
  }

  @override
  Future<List<Map<String, dynamic>>> getInputInfo(String sessionId) {
    return Future.value([
      {
        'name': 'input1',
        'type': 'FLOAT',
        'shape': [1, 3, 224, 224],
      },
      {
        'name': 'input2',
        'type': 'INT64',
        'shape': [1, 10],
      },
    ]);
  }

  @override
  Future<Map<String, dynamic>> getMetadata(String sessionId) {
    return Future.value({
      'producerName': 'ONNX Runtime Test',
      'graphName': 'TestModel',
      'domain': 'test.domain',
      'description': 'Test model for unit tests',
      'version': 1,
      'customMetadataMap': {'key1': 'value1', 'key2': 'value2'},
    });
  }

  @override
  Future<List<Map<String, dynamic>>> getOutputInfo(String sessionId) {
    return Future.value([
      {
        'name': 'output1',
        'type': 'FLOAT',
        'shape': [1, 1000],
      },
      {
        'name': 'output2',
        'type': 'FLOAT',
        'shape': [1, 2, 2],
      },
    ]);
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

  group('OrtSession initialization', () {
    test('fromMap creates session with correct properties', () {
      final map = {
        'sessionId': 'custom_session_id',
        'inputNames': ['custom_input1', 'custom_input2'],
        'outputNames': ['custom_output1', 'custom_output2'],
      };

      final customSession = OrtSession.fromMap(map);

      expect(customSession.id, 'custom_session_id');
      expect(customSession.inputNames, ['custom_input1', 'custom_input2']);
      expect(customSession.outputNames, ['custom_output1', 'custom_output2']);
    });

    test('session created with correct initial values', () {
      expect(session.id, 'test_session_id');
      expect(session.inputNames, ['input1', 'input2']);
      expect(session.outputNames, ['output1', 'output2']);
    });
  });

  group('OrtSession run method', () {
    test('run passes correct parameters to platform', () async {
      final inputs = {
        'input1': [1.0, 2.0, 3.0],
        'input1_shape': [1, 3],
        'input2': [4, 5, 6],
        'input2_shape': [1, 3],
      };

      final runOptions = OrtRunOptions(logSeverityLevel: true, logVerbosityLevel: false, terminate: true);

      await session.run(inputs, options: runOptions);

      // Verify the correct parameters were passed to the platform implementation
      expect(mockPlatform.lastSessionIdForRun, 'test_session_id');
      expect(mockPlatform.lastInputsForRun, inputs);
      expect(mockPlatform.lastRunOptions, runOptions.toMap());
    });

    test('run returns correct output structure', () async {
      final outputs = await session.run({
        'input1': [1.0, 2.0, 3.0],
      });

      expect(outputs, isNotNull);
      expect(outputs.keys, containsAll(['output1', 'output2']));
      expect(outputs['output1'], [1.0, 2.0, 3.0]);
      expect(outputs['output2'], isA<List>());
      expect(outputs['output2'][0], isA<List>());
    });
  });

  group('OrtSession close method', () {
    test('close calls platform implementation with correct session ID', () async {
      await session.close();

      expect(mockPlatform.lastClosedSessionId, 'test_session_id');
    });
  });

  group('OrtSession metadata methods', () {
    test('getMetadata returns properly structured metadata', () async {
      final metadata = await session.getMetadata();

      expect(metadata, isA<OrtModelMetadata>());
      expect(metadata.producerName, 'ONNX Runtime Test');
      expect(metadata.graphName, 'TestModel');
      expect(metadata.domain, 'test.domain');
      expect(metadata.description, 'Test model for unit tests');
      expect(metadata.version, 1);
      expect(metadata.customMetadataMap, {'key1': 'value1', 'key2': 'value2'});
    });

    test('getInputInfo returns input tensor information', () async {
      final inputInfo = await session.getInputInfo();

      expect(inputInfo, isA<List<Map<String, dynamic>>>());
      expect(inputInfo.length, 2);
      expect(inputInfo[0]['name'], 'input1');
      expect(inputInfo[0]['type'], 'FLOAT');
      expect(inputInfo[0]['shape'], [1, 3, 224, 224]);
    });

    test('getOutputInfo returns output tensor information', () async {
      final outputInfo = await session.getOutputInfo();

      expect(outputInfo, isA<List<Map<String, dynamic>>>());
      expect(outputInfo.length, 2);
      expect(outputInfo[0]['name'], 'output1');
      expect(outputInfo[0]['type'], 'FLOAT');
      expect(outputInfo[0]['shape'], [1, 1000]);
    });
  });
}
