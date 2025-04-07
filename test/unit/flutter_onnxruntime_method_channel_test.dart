import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_method_channel.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late MethodChannelFlutterOnnxruntime platform;
  const MethodChannel channel = MethodChannel('flutter_onnxruntime');

  setUp(() {
    platform = MethodChannelFlutterOnnxruntime();
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger.setMockMethodCallHandler(channel, (
      MethodCall methodCall,
    ) async {
      switch (methodCall.method) {
        case 'getPlatformVersion':
          return '42';
        case 'createSession':
          return {
            'sessionId': 'test_session_id',
            'inputNames': ['input1', 'input2'],
            'outputNames': ['output1', 'output2'],
          };
        case 'runInference':
          return {
            'output1': [1.0, 2.0, 3.0],
            'output2': [
              [4.0, 5.0],
              [6.0, 7.0],
            ],
          };
        case 'getMetadata':
          return {
            'producerName': 'Test Producer',
            'graphName': 'Test Graph',
            'domain': 'test.domain',
            'description': 'Test Description',
            'version': 1,
            'customMetadataMap': {'key1': 'value1'},
          };
        case 'getInputInfo':
          return [
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
          ];
        case 'getOutputInfo':
          return [
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
          ];
        case 'closeSession':
          return null;
        default:
          return null;
      }
    });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger.setMockMethodCallHandler(channel, null);
  });

  group('Method channel tests', () {
    test('getPlatformVersion returns correct version', () async {
      expect(await platform.getPlatformVersion(), '42');
    });

    test('createSession returns valid session data', () async {
      final result = await platform.createSession('test_model.onnx');

      expect(result, isNotNull);
      expect(result['sessionId'], 'test_session_id');
      expect(result['inputNames'], ['input1', 'input2']);
      expect(result['outputNames'], ['output1', 'output2']);
    });

    test('runInference returns expected outputs', () async {
      final inputs = {
        'input1': [1.0, 2.0, 3.0],
        'input1_shape': [1, 3],
        'input2': [4, 5, 6],
        'input2_shape': [1, 3],
      };

      final outputs = await platform.runInference('test_session_id', inputs);

      expect(outputs, isNotNull);
      expect(outputs['output1'], [1.0, 2.0, 3.0]);
      expect(outputs['output2'], [
        [4.0, 5.0],
        [6.0, 7.0],
      ]);
    });

    test('getMetadata returns model metadata', () async {
      final metadata = await platform.getMetadata('test_session_id');

      expect(metadata, isNotNull);
      expect(metadata['producerName'], 'Test Producer');
      expect(metadata['graphName'], 'Test Graph');
      expect(metadata['domain'], 'test.domain');
      expect(metadata['description'], 'Test Description');
      expect(metadata['version'], 1);
      expect(metadata['customMetadataMap'], {'key1': 'value1'});
    });

    test('getInputInfo returns input tensor information', () async {
      final inputInfo = await platform.getInputInfo('test_session_id');

      expect(inputInfo, isNotNull);
      expect(inputInfo.length, 2);
      expect(inputInfo[0]['name'], 'input1');
      expect(inputInfo[0]['type'], 'FLOAT');
      expect(inputInfo[0]['shape'], [1, 3, 224, 224]);
    });

    test('getOutputInfo returns output tensor information', () async {
      final outputInfo = await platform.getOutputInfo('test_session_id');

      expect(outputInfo, isNotNull);
      expect(outputInfo.length, 2);
      expect(outputInfo[0]['name'], 'output1');
      expect(outputInfo[0]['type'], 'FLOAT');
      expect(outputInfo[0]['shape'], [1, 1000]);
    });

    test('closeSession completes without error', () async {
      await expectLater(platform.closeSession('test_session_id'), completes);
    });
  });
}
