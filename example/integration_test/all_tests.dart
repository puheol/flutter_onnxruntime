// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Single file for all integration tests
//
// All integration tests are grouped into a single file due to an issue in Linux and macOS reported at:
// https://github.com/flutter/flutter/issues/135673
// Two models used in this Test: Addition model and Transpose-Avg Model
//
// The Addition Model is a simple model which perform Add operation between two tensor A and B, results in C
//
// The Transpose-Avg model operation is defined as follows:
// def forward(self, A, B):
//     # Transpose tensor B (from [batch,n,m] to [batch,m,n])
//     B_transposed = B.transpose(1, 2)
//     # Add transposed B to A
//     summed = A + B_transposed
//     # Multiply element-wise with fixed tensor
//     C = summed * 0.5
//     return C
//
// The model has two inputs: A and B
// A is a tensor with shape [-1, 2, 3]
// B is a tensor with shape [-1, 3, 2]
// The model has one output: C
// C is a tensor with shape [-1, 2, 3]
//
// The model has three versions:
// * FP32: input and output are float32
// * INT32: input and output are int32
// * FP16: model is fp16

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('OrtValue Tests', () {
    group('Round-trip tests', () {
      testWidgets('Float32 round-trip test', (WidgetTester tester) async {
        final inputData = Float32List.fromList([1.1, 2.2, 3.3, 4.4]);
        final shape = [2, 2]; // 2x2 matrix

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.float32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], closeTo(inputData[i], 1e-5));
        }

        await tensor.dispose();
      });

      testWidgets('Int32 round-trip test', (WidgetTester tester) async {
        final inputData = Int32List.fromList([1, 2, 3, 4, 5, 6]);
        final shape = [2, 3]; // 2x3 matrix

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asList();
        expect(retrievedData.length, 6);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], inputData[i]);
        }

        await tensor.dispose();
      });

      testWidgets('Uint8 round-trip test', (WidgetTester tester) async {
        final inputData = Uint8List.fromList([10, 20, 30, 40]);
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.uint8);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], inputData[i]);
        }

        await tensor.dispose();
      });

      testWidgets('Boolean round-trip test', (WidgetTester tester) async {
        final inputData = [true, false, true, false];
        final shape = [2, 2]; // 2x2 matrix

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.bool);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          final retrievedBoolValue = retrievedData[i] == 1;
          expect(retrievedBoolValue, inputData[i]);
        }

        await tensor.dispose();
      });

      testWidgets('Regular List to Float32 conversion test', (WidgetTester tester) async {
        final inputData = [1.1, 2.2, 3.3, 4.4];
        final shape = [2, 2]; // 2x2 matrix

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.float32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], closeTo(inputData[i], 1e-5));
        }

        await tensor.dispose();
      });

      testWidgets('Regular List to Int32 conversion test', (WidgetTester tester) async {
        final inputData = [1, 2, 3, 4];
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], inputData[i]);
        }

        await tensor.dispose();
      });
    });

    group('Type conversion tests', () {
      testWidgets('Float32 to Int32 conversion', (WidgetTester tester) async {
        final inputData = Float32List.fromList([1.1, 2.2, 3.3, 4.4]);
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.float32);

        final convertedTensor = await tensor.to(OrtDataType.int32);
        expect(convertedTensor.dataType, OrtDataType.int32);
        expect(convertedTensor.shape, shape);

        final retrievedData = await convertedTensor.asList();
        expect(retrievedData.length, 4);
        expect(retrievedData[0], 1);
        expect(retrievedData[1], 2);
        expect(retrievedData[2], 3);
        expect(retrievedData[3], 4);

        await tensor.dispose();
        await convertedTensor.dispose();
      });

      testWidgets('Int32 to Float32 conversion', (WidgetTester tester) async {
        final inputData = Int32List.fromList([1, 2, 3, 4]);
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int32);

        final convertedTensor = await tensor.to(OrtDataType.float32);
        expect(convertedTensor.dataType, OrtDataType.float32);
        expect(convertedTensor.shape, shape);

        final retrievedData = await convertedTensor.asList();
        expect(retrievedData.length, 4);
        expect(retrievedData[0], closeTo(1.0, 1e-5));
        expect(retrievedData[1], closeTo(2.0, 1e-5));
        expect(retrievedData[2], closeTo(3.0, 1e-5));
        expect(retrievedData[3], closeTo(4.0, 1e-5));

        await tensor.dispose();
        await convertedTensor.dispose();
      });

      testWidgets('Int32 to Int64 conversion', (WidgetTester tester) async {
        final inputData = Int32List.fromList([1, 2, 3, 4]);
        final shape = [4]; // 1D array

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.int32);

        final convertedTensor = await tensor.to(OrtDataType.int64);
        expect(convertedTensor.dataType, OrtDataType.int64);
        expect(convertedTensor.shape, shape);

        final retrievedData = await convertedTensor.asList();
        expect(retrievedData.length, 4);
        for (int i = 0; i < inputData.length; i++) {
          expect(retrievedData[i], inputData[i]);
        }

        await tensor.dispose();
        await convertedTensor.dispose();
      });

      testWidgets('Same type conversion', (WidgetTester tester) async {
        // same type conversion should clone the tensor to a new tensor
        final inputData = Float32List.fromList([1.1, 2.2]);
        final shape = [2]; // 1D array

        final tensor0 = await OrtValue.fromList(inputData, shape);
        expect(tensor0.dataType, OrtDataType.float32);

        final tensor1 = await tensor0.to(OrtDataType.float32);
        expect(tensor1.dataType, OrtDataType.float32);
        expect(tensor1.shape, shape);

        // release tensor1 but tensor0 should still be valid
        tensor1.dispose();
        expect(tensor0.dataType, OrtDataType.float32);
        expect(tensor0.shape, shape);

        final retrievedData = await tensor0.asList();
        expect(retrievedData.length, 2);
        expect(retrievedData[0], closeTo(1.1, 1e-5));
        expect(retrievedData[1], closeTo(2.2, 1e-5));

        await tensor0.dispose();
      });
    });

    group('Tensor Shape Tests', () {
      testWidgets('Tensor size and target shape mismatch', (WidgetTester tester) async {
        final inputData = [1.1, 2.2, 3.3, 4.4, 5.5];
        final shape = [2, 2];

        // expect to throw an PlatformException
        expect(() async => await OrtValue.fromList(inputData, shape), throwsA(isA<ArgumentError>()));
      });

      testWidgets('Nested list to tensor', (WidgetTester tester) async {
        final inputData = [
          [1.1],
          [2.2, 3.3, 4.4],
        ];
        final shape = [2, 2];

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.float32);
        expect(tensor.shape, shape);

        final retrievedData = await tensor.asList();
        expect(retrievedData.length, 4);
      });

      testWidgets('Negative value in shape test', (WidgetTester tester) async {
        final inputData = [1.1, 2.2, 3.3, 4.4, 5.5];
        final shape = [-1, 2, 2];

        // expect to throw an ArgumentError
        expect(() async => await OrtValue.fromList(inputData, shape), throwsA(isA<ArgumentError>()));
      });
    });

    group('Error handling tests', () {
      testWidgets('Empty list error test', (WidgetTester tester) async {
        final emptyList = [];
        final shape = [0];

        expect(() async => await OrtValue.fromList(emptyList, shape), throwsArgumentError);
      });

      testWidgets('Mismatched shape error test', (WidgetTester tester) async {
        final inputData = [1.0, 2.0, 3.0, 4.0];
        // Total elements in shape doesn't match list length
        final wrongShape = [5, 5];

        expect(() async => await OrtValue.fromList(inputData, wrongShape), throwsA(anything));
      });

      testWidgets('Unsupported conversion test', (WidgetTester tester) async {
        final inputData = [true, false, true];
        final shape = [3];

        final tensor = await OrtValue.fromList(inputData, shape);
        expect(tensor.dataType, OrtDataType.bool);

        // This test checks that converting boolean tensor to float32 either
        // throws an exception or successfully converts the values
        try {
          final convertedTensor = await tensor.to(OrtDataType.float32);
          // If conversion succeeds, verify the data
          expect(convertedTensor.dataType, OrtDataType.float32);
          final retrievedData = await convertedTensor.asList();
          expect(retrievedData.length, 3);
          await convertedTensor.dispose();
        } catch (e) {
          // If an exception is thrown, that's also acceptable
          // as boolean to float conversion might not be supported
          expect(e, isNotNull);
        }

        await tensor.dispose();
      });
    });
  });

  group('Environment setup', () {
    late OnnxRuntime onnxRuntime;

    setUpAll(() async {
      onnxRuntime = OnnxRuntime();
    });

    testWidgets('Get platform version', (WidgetTester tester) async {
      final version = await onnxRuntime.getPlatformVersion();
      // ignore: avoid_print
      print('Platform Version: $version');
      expect(version, isNotNull);
      expect(version!.isNotEmpty, true);
    });

    testWidgets('Available providers', (WidgetTester tester) async {
      try {
        final providers = await onnxRuntime.getAvailableProviders();
        expect(providers, isNotNull);
        expect(providers.isNotEmpty, true);
      } catch (e) {
        fail('Failed to get available providers: $e');
      }
    });
  });

  group('Session Info Tests', () {
    late OnnxRuntime onnxRuntime;
    late OrtSession session;

    setUpAll(() async {
      onnxRuntime = OnnxRuntime();
      try {
        // Load model from assets
        session = await onnxRuntime.createSessionFromAsset('assets/models/addition_model.ort');
      } catch (e) {
        fail('Failed to create session: $e');
      }
    });

    tearDownAll(() async {
      await session.close();
    });

    testWidgets('Get model metadata', (WidgetTester tester) async {
      final metadata = await session.getMetadata();
      expect(metadata, isNotNull);
      expect(metadata.producerName, isNotNull);
    });

    testWidgets('Get input/output info', (WidgetTester tester) async {
      final inputInfo = await session.getInputInfo();
      expect(inputInfo, isNotNull);
      expect(inputInfo.length, 2); // Addition model has inputs A and B

      // Verify the input names
      expect(inputInfo.map((i) => i['name']).toList(), containsAll(['A', 'B']));

      final outputInfo = await session.getOutputInfo();
      expect(outputInfo, isNotNull);
      expect(outputInfo.length, 1); // Addition model has single output C
      expect(outputInfo[0]['name'], 'C');
    });
  });

  group('Inference Tests with Addition Model', () {
    late OnnxRuntime onnxRuntime;
    late OrtSession session;

    setUpAll(() async {
      onnxRuntime = OnnxRuntime();
      try {
        // Load model from assets
        session = await onnxRuntime.createSessionFromAsset('assets/models/addition_model.ort');
      } catch (e) {
        fail('Failed to create session: $e');
      }
    });

    tearDownAll(() async {
      await session.close();
    });

    testWidgets('Add two numbers', (WidgetTester tester) async {
      // Create OrtValue inputs instead of raw arrays
      final inputA = await OrtValue.fromList([3.0], [1]);
      final inputB = await OrtValue.fromList([4.0], [1]);

      final inputs = {'A': inputA, 'B': inputB};

      final outputs = await session.run(inputs);
      final outputTensor = outputs['C'];
      final outputData = await outputTensor!.asList();

      expect(outputs.length, 1);
      expect(outputTensor.dataType, OrtDataType.float32);
      expect(outputTensor.shape, [1]);
      expect(outputData[0], 7.0); // 3 + 4 = 7

      // Clean up
      await inputA.dispose();
      await inputB.dispose();
    });

    testWidgets('Add two arrays of numbers', (WidgetTester tester) async {
      // Create OrtValue inputs instead of raw arrays
      final inputA = await OrtValue.fromList([1.1, 2.2, 3.3], [3]);
      final inputB = await OrtValue.fromList([4.4, 5.5, 6.6], [3]);

      final inputs = {'A': inputA, 'B': inputB};

      final outputs = await session.run(inputs);
      final outputTensor = outputs['C'];
      final outputData = await outputTensor!.asList();

      expect(outputs.length, 1);
      expect(outputTensor.dataType, OrtDataType.float32);
      expect(outputTensor.shape, [3]);
      expect(outputData.length, 3);
      expect(outputData[0], closeTo(5.5, 1e-5)); // 1.1 + 4.4 ≈ 5.5
      expect(outputData[1], closeTo(7.7, 1e-5)); // 2.2 + 5.5 ≈ 7.7
      expect(outputData[2], closeTo(9.9, 1e-5)); // 3.3 + 6.6 ≈ 9.9

      // Clean up
      await inputA.dispose();
      await inputB.dispose();
    });

    testWidgets('Run inference with run options', (WidgetTester tester) async {
      final runOptions = OrtRunOptions(logSeverityLevel: 1, logVerbosityLevel: 1, terminate: false);
      final inputs = {
        'A': await OrtValue.fromList([3.0], [1]),
        'B': await OrtValue.fromList([4.0], [1]),
      };
      final outputs = await session.run(inputs, options: runOptions);
      final outputTensor = outputs['C'];
      final outputData = await outputTensor!.asList();

      expect(outputs.length, 1);
      expect(outputTensor.dataType, OrtDataType.float32);
      expect(outputTensor.shape, [1]);
      expect(outputData[0], 7.0); // 3 + 4 = 7

      // Clean up
      await inputs['A']!.dispose();
      await inputs['B']!.dispose();
      await outputs['C']!.dispose();
    });

    testWidgets('Run inference with run options and terminate', (WidgetTester tester) async {
      // Skip the test for iOS and macOS as terminate is not supported
      if (Platform.isIOS || Platform.isMacOS) {
        return; // Skip the test
      }
      final runOptions = OrtRunOptions(logSeverityLevel: 1, logVerbosityLevel: 1, terminate: true);
      final inputs = {
        'A': await OrtValue.fromList([3.0], [1]),
        'B': await OrtValue.fromList([4.0], [1]),
      };

      // expect the run to throw an exception since terminate is true
      expect(() async {
        await session.run(inputs, options: runOptions);
      }, throwsA(isA<Exception>()));

      // Clean up
      await inputs['A']!.dispose();
      await inputs['B']!.dispose();
    });
  });

  group('Transpose and Avg Model Tests', () {
    group('FP32 model test', () {
      late OnnxRuntime onnxRuntime;
      late OrtSession session;
      late Map<String, OrtValue> inputs;

      setUpAll(() async {
        onnxRuntime = OnnxRuntime();
        session = await onnxRuntime.createSessionFromAsset('assets/models/transpose_and_avg_model_fp32.onnx');
      });

      tearDownAll(() async {
        await session.close();
      });

      testWidgets('FP32 model inference test', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 2, 3]),
          'B': await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1, 3, 2]),
        };
        final outputs = await session.run(inputs);
        final output = outputs['C'];
        expect(output!.dataType, OrtDataType.float32);
        expect(output.shape, [1, 2, 3]);
        final outputData = await output.asList();
        expect(outputData.length, 6);
        expect(outputData.every((e) => e == 1.5), true);

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
        await output.dispose();
      });

      testWidgets('FP32 model inference test with multi-batch', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList(
            [
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [2, 2, 3],
          ),
          'B': await OrtValue.fromList(
            [
              [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ],
            [2, 3, 2],
          ),
        };
        final outputs = await session.run(inputs);
        final output = outputs['C'];
        expect(output!.dataType, OrtDataType.float32);
        expect(output.shape, [2, 2, 3]);
        final outputData = await output.asList();
        expect(outputData.length, 12);
        expect(outputData.every((e) => e == 1.5), true);

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
        await output.dispose();
      });

      testWidgets('Invalid input rank test', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2, 3]),
          'B': await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [3, 2]),
        };
        // expect to throw an exeption
        expect(() async => await session.run(inputs), throwsA(isA<Exception>()));

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
      });

      testWidgets('Invalid input shape test', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 2, 3]),
          'B': await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1, 2, 3]),
        };
        // expect to throw an exeption
        expect(() async => await session.run(inputs), throwsA(isA<Exception>()));

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
      });

      testWidgets('Invalid input type', (WidgetTester tester) async {
        // create int tensors
        final tensorA = await OrtValue.fromList([1, 1, 1, 1, 1, 1], [1, 2, 3]);
        final tensorB = await OrtValue.fromList([2, 2, 2, 2, 2, 2], [1, 3, 2]);
        // expect to throw an exeption
        expect(() async => await session.run({'A': tensorA, 'B': tensorB}), throwsA(isA<Exception>()));

        // clean up
        await tensorA.dispose();
        await tensorB.dispose();
      });
    });

    group('INT32 model test', () {
      late OnnxRuntime onnxRuntime;
      late OrtSession session;
      late Map<String, OrtValue> inputs;

      setUpAll(() async {
        onnxRuntime = OnnxRuntime();
        session = await onnxRuntime.createSessionFromAsset('assets/models/transpose_and_avg_model_int32.onnx');
      });

      tearDownAll(() async {
        await session.close();
      });

      testWidgets('INT32 model inference test', (WidgetTester tester) async {
        inputs = {
          'A': await OrtValue.fromList([1, 1, 1, 1, 1, 1], [1, 2, 3]),
          'B': await OrtValue.fromList([2, 2, 2, 2, 2, 2], [1, 3, 2]),
        };
        final outputs = await session.run(inputs);
        final output = outputs['C'];
        expect(output!.dataType, OrtDataType.int32);
        expect(output.shape, [1, 2, 3]);

        final outputData = await output.asList();
        expect(outputData.length, 6);
        expect(outputData.every((e) => e == 1), true); // 1 + 2 = 3, 3 * 0.5 = 1.5 -> 1

        // clean up
        for (var input in inputs.values) {
          input.dispose();
        }
        await output.dispose();
      });
    });

    group('FP16 model test', () {
      late OnnxRuntime onnxRuntime;
      late OrtSession session;

      setUpAll(() async {
        onnxRuntime = OnnxRuntime();
        session = await onnxRuntime.createSessionFromAsset('assets/models/transpose_and_avg_model_fp16.onnx');
      });

      tearDownAll(() async {
        await session.close();
      });

      testWidgets('FP16 model inference test', (WidgetTester tester) async {
        final tensorA = await OrtValue.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 2, 3]);
        final tensorB = await OrtValue.fromList([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1, 3, 2]);
        // only support Android
        if (Platform.isAndroid) {
          // convert to fp16
          final tensorAFp16 = await tensorA.to(OrtDataType.float16);
          final tensorBFp16 = await tensorB.to(OrtDataType.float16);

          final outputs = await session.run({'A': tensorAFp16, 'B': tensorBFp16});
          final output = outputs['C'];
          expect(output!.dataType, OrtDataType.float16);
          expect(output.shape, [1, 2, 3]);

          final outputData = await output.asList();
          expect(outputData.length, 6);
          expect(outputData.every((e) => e == 1.5), true);

          // clean up
          await tensorA.dispose();
          await tensorB.dispose();
          await tensorAFp16.dispose();
          await tensorBFp16.dispose();
          await output.dispose();
        }
        // TODO: Standardize behaviour across platforms
        else if (Platform.isIOS || Platform.isMacOS) {
          expect(() async => await tensorA.to(OrtDataType.float16), throwsA(isA<PlatformException>()));
        } else {
          // not possible to create a fp16 tensor on non-Android platforms
          expect(() async => await tensorA.to(OrtDataType.float16), throwsA(isA<TypeError>()));
        }
      });
    });
  });
}
