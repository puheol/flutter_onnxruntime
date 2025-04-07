import 'dart:typed_data';

import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';

/// Represents a data type in ONNX Runtime
enum OrtDataType {
  // Numeric types
  float32,
  float16,
  int32,
  int64,
  int16,
  int8,
  uint8,
  uint16,
  uint32,
  uint64,
  bool,

  // String type
  string,

  // Complex types
  complex64,
  complex128,

  // Other numeric types
  bfloat16,
}

/// Represents a device in ONNX Runtime
enum OrtDevice { cpu, cuda, tensorrt, openVINO }

/// OrtValue represents a tensor or other data structure used for input/output in ONNX Runtime.
///
/// This class manages memory for tensor data and provides methods for data type conversion
/// and device transfer. It wraps the native OrtValue (C/C++) or OnnxTensor (Java) types from
/// the ONNX Runtime API.
class OrtValue {
  /// Unique identifier for this tensor in the native code
  final String id;

  /// Data type of this tensor
  final OrtDataType dataType;

  /// Shape of the tensor as a list of dimensions
  final List<int> shape;

  /// Device where the tensor is stored
  final OrtDevice device;

  /// Private constructor
  OrtValue._({required this.id, required this.dataType, required this.shape, this.device = OrtDevice.cpu});

  /// Creates an OrtValue from a map returned by the platform interface
  factory OrtValue.fromMap(Map<String, dynamic> map) {
    return OrtValue._(
      id: map['valueId'] as String,
      dataType: OrtDataType.values.firstWhere(
        (dt) => dt.toString() == 'OrtDataType.${map['dataType']}',
        orElse: () => OrtDataType.float32,
      ),
      shape: List<int>.from(map['shape'] ?? []),
      device: OrtDevice.values.firstWhere(
        (dev) => dev.toString() == 'OrtDevice.${map['device']}',
        orElse: () => OrtDevice.cpu,
      ),
    );
  }

  /// Creates an OrtValue from a Float32List
  ///
  /// [data] is the data to create the tensor from
  /// [shape] is the shape of the tensor
  /// [targetType] is the optional target data type (defaults to float32)
  /// [device] is the optional target device (defaults to CPU)
  static Future<OrtValue> fromFloat32List(
    Float32List data,
    List<int> shape, {
    OrtDataType targetType = OrtDataType.float32,
    OrtDevice device = OrtDevice.cpu,
  }) async {
    final result = await FlutterOnnxruntimePlatform.instance.createOrtValue(
      'float32',
      data,
      shape,
      targetType.toString().split('.').last,
      device.toString().split('.').last,
    );
    return OrtValue.fromMap(result);
  }

  /// Creates an OrtValue from an Int32List
  ///
  /// [data] is the data to create the tensor from
  /// [shape] is the shape of the tensor
  /// [targetType] is the optional target data type (defaults to int32)
  /// [device] is the optional target device (defaults to CPU)
  static Future<OrtValue> fromInt32List(
    Int32List data,
    List<int> shape, {
    OrtDataType targetType = OrtDataType.int32,
    OrtDevice device = OrtDevice.cpu,
  }) async {
    final result = await FlutterOnnxruntimePlatform.instance.createOrtValue(
      'int32',
      data,
      shape,
      targetType.toString().split('.').last,
      device.toString().split('.').last,
    );
    return OrtValue.fromMap(result);
  }

  /// Creates an OrtValue from an Int64List
  ///
  /// [data] is the data to create the tensor from
  /// [shape] is the shape of the tensor
  /// [targetType] is the optional target data type (defaults to int64)
  /// [device] is the optional target device (defaults to CPU)
  static Future<OrtValue> fromInt64List(
    Int64List data,
    List<int> shape, {
    OrtDataType targetType = OrtDataType.int64,
    OrtDevice device = OrtDevice.cpu,
  }) async {
    final result = await FlutterOnnxruntimePlatform.instance.createOrtValue(
      'int64',
      data,
      shape,
      targetType.toString().split('.').last,
      device.toString().split('.').last,
    );
    return OrtValue.fromMap(result);
  }

  /// Creates an OrtValue from a Uint8List
  ///
  /// [data] is the data to create the tensor from
  /// [shape] is the shape of the tensor
  /// [targetType] is the optional target data type (defaults to uint8)
  /// [device] is the optional target device (defaults to CPU)
  static Future<OrtValue> fromUint8List(
    Uint8List data,
    List<int> shape, {
    OrtDataType targetType = OrtDataType.uint8,
    OrtDevice device = OrtDevice.cpu,
  }) async {
    final result = await FlutterOnnxruntimePlatform.instance.createOrtValue(
      'uint8',
      data,
      shape,
      targetType.toString().split('.').last,
      device.toString().split('.').last,
    );
    return OrtValue.fromMap(result);
  }

  /// Creates an OrtValue from a Bool list
  ///
  /// [data] is the data to create the tensor from
  /// [shape] is the shape of the tensor
  /// [device] is the optional target device (defaults to CPU)
  static Future<OrtValue> fromBoolList(List<bool> data, List<int> shape, {OrtDevice device = OrtDevice.cpu}) async {
    final result = await FlutterOnnxruntimePlatform.instance.createOrtValue(
      'bool',
      data,
      shape,
      'bool',
      device.toString().split('.').last,
    );
    return OrtValue.fromMap(result);
  }

  /// Convert this tensor to a different data type
  ///
  /// [targetType] is the target data type to convert to
  Future<OrtValue> to(OrtDataType targetType) async {
    final result = await FlutterOnnxruntimePlatform.instance.convertOrtValue(id, targetType.toString().split('.').last);
    return OrtValue.fromMap(result);
  }

  /// Move this tensor to a different device
  ///
  /// [targetDevice] is the target device to move to
  Future<OrtValue> toDevice(OrtDevice targetDevice) async {
    final result = await FlutterOnnxruntimePlatform.instance.moveOrtValueToDevice(
      id,
      targetDevice.toString().split('.').last,
    );
    return OrtValue.fromMap(result);
  }

  /// Get the data from this tensor as a Float32List
  ///
  /// This method will convert the tensor to float32 if it's not already
  Future<Float32List> asFloat32List() async {
    final data = await FlutterOnnxruntimePlatform.instance.getOrtValueData(id, 'float32');
    return Float32List.fromList(List<double>.from(data['data']));
  }

  /// Get the data from this tensor as an Int32List
  ///
  /// This method will convert the tensor to int32 if it's not already
  Future<Int32List> asInt32List() async {
    final data = await FlutterOnnxruntimePlatform.instance.getOrtValueData(id, 'int32');
    return Int32List.fromList(List<int>.from(data['data']));
  }

  /// Get the data from this tensor as an Int64List
  ///
  /// This method will convert the tensor to int64 if it's not already
  Future<Int64List> asInt64List() async {
    final data = await FlutterOnnxruntimePlatform.instance.getOrtValueData(id, 'int64');
    return Int64List.fromList(List<int>.from(data['data']));
  }

  /// Get the data from this tensor as a Uint8List
  ///
  /// This method will convert the tensor to uint8 if it's not already
  Future<Uint8List> asUint8List() async {
    final data = await FlutterOnnxruntimePlatform.instance.getOrtValueData(id, 'uint8');
    return Uint8List.fromList(List<int>.from(data['data']));
  }

  /// Get the data from this tensor as a list of booleans
  ///
  /// This method will convert the tensor to boolean if it's not already
  Future<List<bool>> asBoolList() async {
    final data = await FlutterOnnxruntimePlatform.instance.getOrtValueData(id, 'bool');
    return List<bool>.from(data['data']);
  }

  /// Release native resources associated with this tensor
  Future<void> dispose() async {
    await FlutterOnnxruntimePlatform.instance.releaseOrtValue(id);
  }
}
