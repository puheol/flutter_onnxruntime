import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterOnnxruntimePlatform with MockPlatformInterfaceMixin implements FlutterOnnxruntimePlatform {
  // Tracks method calls for verification
  String? lastSourceType;
  dynamic lastSourceData;
  List<int>? lastShape;
  String? lastValueIdForConversion;
  String? lastValueIdForRelease;
  String? lastValueIdForData;
  String? lastRequestedDataType;

  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) {
    return Future.value({
      'sessionId': 'test_session_id',
      'inputNames': ['input1'],
      'outputNames': ['output1'],
    });
  }

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, dynamic> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    return Future.value({
      'outputs': {
        'output1': [1.0, 2.0, 3.0, 4.0],
        'output1_shape': [2, 2],
      },
    });
  }

  @override
  Future<Map<String, dynamic>> createOrtValue(String sourceType, dynamic data, List<int> shape) {
    // Track the call parameters
    lastSourceType = sourceType;
    lastSourceData = data;
    lastShape = shape;

    // Return a mock OrtValue map
    String dataType = sourceType;

    return Future.value({
      'valueId': 'test_value_id_${DateTime.now().millisecondsSinceEpoch}',
      'dataType': dataType,
      'shape': shape,
      'device': 'cpu',
    });
  }

  @override
  Future<Map<String, dynamic>> convertOrtValue(String valueId, String targetType) {
    // Track the call
    lastValueIdForConversion = valueId;

    return Future.value({
      'valueId': 'converted_$valueId',
      'dataType': targetType,
      'shape': [2, 2],
      'device': 'cpu',
    });
  }

  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId, String dataType) {
    // Track the call
    lastValueIdForData = valueId;
    lastRequestedDataType = dataType;

    // Return different data based on the requested type
    switch (dataType) {
      case 'float32':
        return Future.value({
          'data': [1.0, 2.0, 3.0, 4.0],
          'shape': [2, 2],
        });
      case 'int32':
        return Future.value({
          'data': [1, 2, 3, 4],
          'shape': [2, 2],
        });
      case 'int64':
        return Future.value({
          'data': [1, 2, 3, 4],
          'shape': [2, 2],
        });
      case 'uint8':
        return Future.value({
          'data': [1, 2, 3, 4],
          'shape': [2, 2],
        });
      case 'bool':
        return Future.value({
          'data': [true, false, true, false],
          'shape': [2, 2],
        });
      default:
        return Future.value({
          'data': [],
          'shape': [0],
        });
    }
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
  Future<void> releaseOrtValue(String valueId) {
    lastValueIdForRelease = valueId;
    return Future.value();
  }

  @override
  Future<void> closeSession(String sessionId) => Future.value();

  @override
  Future<Map<String, dynamic>> getMetadata(String sessionId) {
    return Future.value({});
  }

  @override
  Future<List<Map<String, dynamic>>> getInputInfo(String sessionId) {
    return Future.value([]);
  }

  @override
  Future<List<Map<String, dynamic>>> getOutputInfo(String sessionId) {
    return Future.value([]);
  }
}

void main() {
  late MockFlutterOnnxruntimePlatform mockPlatform;
  final FlutterOnnxruntimePlatform initialPlatform = FlutterOnnxruntimePlatform.instance;

  setUp(() {
    mockPlatform = MockFlutterOnnxruntimePlatform();
    FlutterOnnxruntimePlatform.instance = mockPlatform;
  });

  tearDown(() {
    FlutterOnnxruntimePlatform.instance = initialPlatform;
  });

  group('OrtValue creation with fromList', () {
    test('fromList with Float32List should create OrtValue correctly', () async {
      final testData = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final testShape = [2, 2];

      final tensor = await OrtValue.fromList(testData, testShape);

      // Verify correct parameter forwarding
      expect(mockPlatform.lastSourceType, 'float32');
      expect(mockPlatform.lastSourceData, testData);
      expect(mockPlatform.lastShape, testShape);

      // Verify OrtValue properties
      expect(tensor.shape, testShape);
      expect(tensor.dataType, OrtDataType.float32);
      expect(tensor.device, OrtDevice.cpu);
      expect(tensor.id, isNotEmpty);
    });

    test('fromList with Int32List should create OrtValue correctly', () async {
      final testData = Int32List.fromList([1, 2, 3, 4]);
      final testShape = [2, 2];

      final tensor = await OrtValue.fromList(testData, testShape);

      // Verify correct parameter forwarding
      expect(mockPlatform.lastSourceType, 'int32');
      expect(mockPlatform.lastSourceData, testData);
      expect(mockPlatform.lastShape, testShape);

      // Verify OrtValue properties
      expect(tensor.shape, testShape);
      expect(tensor.dataType, OrtDataType.int32);
    });

    test('fromList with Int64List should create OrtValue correctly', () async {
      final testData = Int64List.fromList([1, 2, 3, 4]);
      final testShape = [2, 2];

      final tensor = await OrtValue.fromList(testData, testShape);

      // Verify correct parameter forwarding
      expect(mockPlatform.lastSourceType, 'int64');
      expect(mockPlatform.lastSourceData, testData);
      expect(mockPlatform.lastShape, testShape);

      // Verify OrtValue properties
      expect(tensor.shape, testShape);
      expect(tensor.dataType, OrtDataType.int64);
    });

    test('fromList with Uint8List should create OrtValue correctly', () async {
      final testData = Uint8List.fromList([1, 2, 3, 4]);
      final testShape = [2, 2];

      final tensor = await OrtValue.fromList(testData, testShape);

      // Verify correct parameter forwarding
      expect(mockPlatform.lastSourceType, 'uint8');
      expect(mockPlatform.lastSourceData, testData);
      expect(mockPlatform.lastShape, testShape);

      // Verify OrtValue properties
      expect(tensor.shape, testShape);
      expect(tensor.dataType, OrtDataType.uint8);
    });

    test('fromList with List<bool> should create OrtValue correctly', () async {
      final testData = [true, false, true, false];
      final testShape = [2, 2];

      final tensor = await OrtValue.fromList(testData, testShape);

      // Verify correct parameter forwarding
      expect(mockPlatform.lastSourceType, 'bool');
      expect(mockPlatform.lastSourceData, testData);
      expect(mockPlatform.lastShape, testShape);

      // Verify OrtValue properties
      expect(tensor.shape, testShape);
      expect(tensor.dataType, OrtDataType.bool);
    });

    test('fromList with List<double> should convert to Float32List', () async {
      final testData = [1.0, 2.0, 3.0, 4.0];
      final testShape = [2, 2];

      final tensor = await OrtValue.fromList(testData, testShape);

      // Verify correct parameter forwarding
      expect(mockPlatform.lastSourceType, 'float32');
      expect(mockPlatform.lastSourceData, isA<Float32List>());
      expect(mockPlatform.lastShape, testShape);

      // Verify OrtValue properties
      expect(tensor.shape, testShape);
      expect(tensor.dataType, OrtDataType.float32);
    });

    test('fromList with List<int> within int32 range should convert to Int32List', () async {
      final testData = [1, 2, 3, 4];
      final testShape = [2, 2];

      final tensor = await OrtValue.fromList(testData, testShape);

      // Verify correct parameter forwarding
      expect(mockPlatform.lastSourceType, 'int32');
      expect(mockPlatform.lastSourceData, isA<Int32List>());
      expect(mockPlatform.lastShape, testShape);

      // Verify OrtValue properties
      expect(tensor.shape, testShape);
      expect(tensor.dataType, OrtDataType.int32);
    });

    test('fromList with List<int> exceeding int32 range should convert to Int64List', () async {
      final testData = [1, 2147483648, 3, 4]; // 2147483648 exceeds int32 range
      final testShape = [2, 2];

      final tensor = await OrtValue.fromList(testData, testShape);

      // Verify correct parameter forwarding
      expect(mockPlatform.lastSourceType, 'int64');
      expect(mockPlatform.lastSourceData, isA<Int64List>());
      expect(mockPlatform.lastShape, testShape);

      // Verify OrtValue properties
      expect(tensor.shape, testShape);
      expect(tensor.dataType, OrtDataType.int64);
    });

    test('fromList with List<num> with mixed integers and decimals should convert to Float32List', () async {
      final testData = [1, 2, 3.5, 4];
      final testShape = [2, 2];

      final tensor = await OrtValue.fromList(testData, testShape);

      // Verify correct parameter forwarding
      expect(mockPlatform.lastSourceType, 'float32');
      expect(mockPlatform.lastSourceData, isA<Float32List>());
      expect(mockPlatform.lastShape, testShape);

      // Verify OrtValue properties
      expect(tensor.shape, testShape);
      expect(tensor.dataType, OrtDataType.float32);
    });

    test('fromList with empty list should throw ArgumentError', () async {
      final testData = [];
      final testShape = [0];

      expect(() => OrtValue.fromList(testData, testShape), throwsArgumentError);
    });

    test('fromList with unsupported type should throw ArgumentError', () async {
      final testData = ['string1', 'string2'];
      final testShape = [2];

      expect(() => OrtValue.fromList(testData, testShape), throwsArgumentError);
    });
  });

  group('OrtValue conversion', () {
    test('to() should convert to a different data type', () async {
      // Create an OrtValue first
      final tensor = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      // Convert to a different type
      final convertedTensor = await tensor.to(OrtDataType.int32);

      // Verify the conversion call
      expect(mockPlatform.lastValueIdForConversion, tensor.id);

      // Verify the new tensor
      expect(convertedTensor.id, isNot(tensor.id));
      expect(convertedTensor.dataType, OrtDataType.int32);
      expect(convertedTensor.shape, [2, 2]);
    });

    test('toDevice() should move tensor to a different device', () async {
      // Create an OrtValue first
      final tensor = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      // Move to a different device
      final deviceTensor = await tensor.toDevice(OrtDevice.cuda);

      // Verify properties
      expect(deviceTensor.device, OrtDevice.cuda);
    });
  });

  group('OrtValue data extraction', () {
    test('asFloat32List() should return Float32List data', () async {
      // Create an OrtValue
      final tensor = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      // Get data as Float32List
      final data = await tensor.asFloat32List();

      // Verify the call
      expect(mockPlatform.lastValueIdForData, tensor.id);
      expect(mockPlatform.lastRequestedDataType, 'float32');

      // Verify the returned data
      expect(data, isA<Float32List>());
      expect(data.length, 4);
      expect(data[0], 1.0);
      expect(data[3], 4.0);
    });

    test('asInt32List() should return Int32List data', () async {
      // Create an OrtValue
      final tensor = await OrtValue.fromList(Int32List.fromList([1, 2, 3, 4]), [2, 2]);

      // Get data as Int32List
      final data = await tensor.asInt32List();

      // Verify the call
      expect(mockPlatform.lastValueIdForData, tensor.id);
      expect(mockPlatform.lastRequestedDataType, 'int32');

      // Verify the returned data
      expect(data, isA<Int32List>());
      expect(data.length, 4);
      expect(data[0], 1);
      expect(data[3], 4);
    });

    test('asBoolList() should return bool list data', () async {
      // Create an OrtValue
      final tensor = await OrtValue.fromList([true, false, true, false], [2, 2]);

      // Get data as bool list
      final data = await tensor.asBoolList();

      // Verify the call
      expect(mockPlatform.lastValueIdForData, tensor.id);
      expect(mockPlatform.lastRequestedDataType, 'bool');

      // Verify the returned data
      expect(data, isA<List<bool>>());
      expect(data.length, 4);
      expect(data[0], true);
      expect(data[1], false);
    });
  });

  group('OrtValue memory management', () {
    test('dispose() should release native resources', () async {
      // Create an OrtValue
      final tensor = await OrtValue.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [2, 2]);

      final valueId = tensor.id;

      // Dispose the tensor
      await tensor.dispose();

      // Verify release was called with the correct ID
      expect(mockPlatform.lastValueIdForRelease, valueId);
    });
  });

  group('OrtValue creation from map', () {
    test('fromMap should create OrtValue with correct properties', () {
      final map = {
        'valueId': 'test_id',
        'dataType': 'float32',
        'shape': [2, 2],
        'device': 'cpu',
      };

      final tensor = OrtValue.fromMap(map);

      expect(tensor.id, 'test_id');
      expect(tensor.dataType, OrtDataType.float32);
      expect(tensor.shape, [2, 2]);
      expect(tensor.device, OrtDevice.cpu);
    });

    test('fromMap should handle missing or invalid properties', () {
      final map = {
        'valueId': 'test_id',
        // Missing dataType
        'shape': [], // Empty shape
        'device': 'invalid_device',
      };

      final tensor = OrtValue.fromMap(map);

      expect(tensor.id, 'test_id');
      expect(tensor.dataType, OrtDataType.float32); // Default
      expect(tensor.shape, isEmpty);
      expect(tensor.device, OrtDevice.cpu); // Default
    });
  });
}
