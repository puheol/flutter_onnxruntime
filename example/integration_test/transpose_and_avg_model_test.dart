// This test is used to test the transpose and avg model
// The model operation is defined as follows:
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

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

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
        } else {
          // not possible to create a fp16 tensor on non-Android platforms
          expect(() async => await tensorA.to(OrtDataType.float16), throwsA(isA<TypeError>()));
        }
      });
    });
  });
}
