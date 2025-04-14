import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'dart:typed_data';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('OrtValue Integration Tests', () {
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
}
