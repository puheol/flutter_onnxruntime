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

  /// Creates an OrtValue from any supported list type
  ///
  /// This method detects the list type and converts it to the appropriate format.
  // ignore: unintended_html_in_doc_comment
  /// Supported types include Float32List, Int32List, Int64List, Uint8List, List<bool>,
  // ignore: unintended_html_in_doc_comment
  /// and their corresponding Dart List<num> types. The default device is cpu.
  ///
  /// [data] is the data to create the tensor from (any supported list type)
  /// [shape] is the shape of the tensor
  static Future<OrtValue> fromList(dynamic data, List<int> shape) async {
    // If data is a regular List, convert it to the appropriate TypedData
    if (data is List && !(data is Float32List || data is Int32List || data is Int64List || data is Uint8List)) {
      data = _convertListToTypedData(data);
    }

    String sourceType;

    if (data is Float32List) {
      sourceType = 'float32';
    } else if (data is Int32List) {
      sourceType = 'int32';
    } else if (data is Int64List) {
      sourceType = 'int64';
    } else if (data is Uint8List) {
      sourceType = 'uint8';
    } else if (data is List<bool>) {
      sourceType = 'bool';
    } else {
      throw ArgumentError('Unsupported data type: ${data.runtimeType}');
    }

    final result = await FlutterOnnxruntimePlatform.instance.createOrtValue(sourceType, data, shape);
    return OrtValue.fromMap(result);
  }

  /// Converts a regular List to appropriate TypedData based on content
  static dynamic _convertListToTypedData(List data) {
    if (data.isEmpty) {
      throw ArgumentError('Cannot create OrtValue from empty list');
    }

    final firstElement = data.first;

    if (firstElement is bool) {
      // Just return List<bool> as is, it's handled separately
      return data.cast<bool>();
    } else if (firstElement is double ||
        (firstElement is num && data.any((e) => e is double || (e is num && e % 1 != 0)))) {
      // Convert to Float32List if any element is a double or has decimal part
      final typedList = Float32List(data.length);
      for (int i = 0; i < data.length; i++) {
        typedList[i] = (data[i] as num).toDouble();
      }
      return typedList;
    } else if (firstElement is int || firstElement is num) {
      // Check if int64 is needed
      bool needsInt64 = false;
      for (var item in data) {
        int value = (item as num).toInt();
        if (value > 2147483647 || value < -2147483648) {
          needsInt64 = true;
          break;
        }
      }

      if (needsInt64) {
        final typedList = Int64List(data.length);
        for (int i = 0; i < data.length; i++) {
          typedList[i] = (data[i] as num).toInt();
        }
        return typedList;
      } else {
        final typedList = Int32List(data.length);
        for (int i = 0; i < data.length; i++) {
          typedList[i] = (data[i] as num).toInt();
        }
        return typedList;
      }
    }

    throw ArgumentError('Unsupported element type: ${firstElement.runtimeType} in list');
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
