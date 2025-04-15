// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
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
            'output1': [
              'output_value_1',
              'float32',
              [3],
            ],
            'output2': [
              'output_value_2',
              'float32',
              [2, 2],
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
        case 'getAvailableProviders':
          return ['CPU', 'CUDA', 'CoreML'];
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

    test('runInference with OrtValues returns expected outputs', () async {
      // Create mock OrtValue by using the fromMap constructor
      final ortValue1 = OrtValue.fromMap({
        'valueId': 'test_value_1',
        'dataType': 'float32',
        'shape': [1, 3],
      });

      final ortValue2 = OrtValue.fromMap({
        'valueId': 'test_value_2',
        'dataType': 'float32',
        'shape': [1, 3],
      });

      final inputs = {'input1': ortValue1, 'input2': ortValue2};

      final outputs = await platform.runInference('test_session_id', inputs);

      expect(outputs, isNotNull);
      expect(outputs, isA<Map<String, dynamic>>());
      expect(outputs.keys, containsAll(['output1', 'output2']));

      // Each output should contain values needed to create an OrtValue
      expect(outputs['output1'], isA<List>());
      expect(outputs['output1'].length, 3); // valueId, dataType, shape
      expect(outputs['output1'][0], 'output_value_1');
      expect(outputs['output1'][1], 'float32');
      expect(outputs['output1'][2], [3]);

      expect(outputs['output2'], isA<List>());
      expect(outputs['output2'][0], 'output_value_2');
      expect(outputs['output2'][1], 'float32');
      expect(outputs['output2'][2], [2, 2]);
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

    test('runInference processes OrtValue inputs correctly', () async {
      // Set up a mock implementation for the method channel
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger.setMockMethodCallHandler(channel, (
        MethodCall methodCall,
      ) async {
        // Verify methodCall.arguments for OrtValue format
        if (methodCall.method == 'runInference') {
          final args = methodCall.arguments as Map<Object?, Object?>;
          final inputs = args['inputs'] as Map<Object?, Object?>;

          // Check that the inputs contain valueId maps
          expect(inputs['input1'], isA<Map>());
          expect((inputs['input1'] as Map)['valueId'], 'test_value_1');

          expect(inputs['input2'], isA<Map>());
          expect((inputs['input2'] as Map)['valueId'], 'test_value_2');

          return {
            'output1': [
              'output_value_1',
              'float32',
              [3],
            ],
            'output2': [
              'output_value_2',
              'float32',
              [2, 2],
            ],
          };
        }
        return null;
      });

      // Create mock OrtValues
      final ortValue1 = OrtValue.fromMap({
        'valueId': 'test_value_1',
        'dataType': 'float32',
        'shape': [1, 3],
      });

      final ortValue2 = OrtValue.fromMap({
        'valueId': 'test_value_2',
        'dataType': 'float32',
        'shape': [1, 3],
      });

      // Run inference with the mock OrtValues
      final result = await platform.runInference('test_session_id', {'input1': ortValue1, 'input2': ortValue2});

      // Verify the result
      expect(result, isA<Map<String, dynamic>>());
      expect(result['output1'], isA<List>());
      expect(result['output1'][0], 'output_value_1');
      expect(result['output1'][1], 'float32');
      expect(result['output1'][2], [3]);
    });

    test('getAvailableProviders returns list of providers', () async {
      // Set up a mock implementation for the method channel
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger.setMockMethodCallHandler(channel, (
        MethodCall methodCall,
      ) async {
        if (methodCall.method == 'getAvailableProviders') {
          return ['CPU', 'CUDA', 'CoreML'];
        }
        return null;
      });

      final providers = await platform.getAvailableProviders();

      expect(providers, isNotNull);
      expect(providers, isA<List<String>>());
      expect(providers.length, 3);
      expect(providers, containsAll(['CPU', 'CUDA', 'CoreML']));
    });
  });
}
