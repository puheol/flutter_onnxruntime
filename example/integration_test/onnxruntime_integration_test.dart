// ONNX Runtime integration test
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

void main() {
  // Initialize integration test binding
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Environment setup', () {
    late OnnxRuntime onnxRuntime;

    setUpAll(() async {
      onnxRuntime = OnnxRuntime();
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
  group('ONNX Runtime Integration Tests', () {
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

    testWidgets('Get platform version', (WidgetTester tester) async {
      final version = await onnxRuntime.getPlatformVersion();
      // ignore: avoid_print
      print('Platform Version: $version');
      expect(version, isNotNull);
      expect(version!.isNotEmpty, true);
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
  });
}
