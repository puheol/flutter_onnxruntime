import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'flutter_onnxruntime_platform_interface.dart';

/// An implementation of [FlutterOnnxruntimePlatform] that uses method channels.
class MethodChannelFlutterOnnxruntime extends FlutterOnnxruntimePlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('flutter_onnxruntime');

  @override
  Future<String?> getPlatformVersion() async {
    return await methodChannel.invokeMethod<String>('getPlatformVersion');
  }

  /// Creates a new session for the given model path.
  ///
  /// [modelPath] is the path to the ONNX model file.
  /// [sessionOptions] is an optional map of session options.
  ///
  /// Returns a map of the session options.
  ///
  /// Using model path allows the native code to load the model directly,
  /// which is more memory efficient as it avoids copying the entire model
  /// through the method channel.
  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) async {
    final result = await methodChannel.invokeMethod<Map<Object?, Object?>>('createSession', {
      'modelPath': modelPath,
      'sessionOptions': sessionOptions ?? {},
    });
    return _convertMapToStringDynamic(result ?? {});
  }

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, dynamic> inputs, {
    Map<String, dynamic>? runOptions,
  }) async {
    final result = await methodChannel.invokeMethod<Map<Object?, Object?>>('runInference', {
      'sessionId': sessionId,
      'inputs': inputs,
      'runOptions': runOptions ?? {},
    });
    return _convertMapToStringDynamic(result ?? {});
  }

  @override
  Future<void> closeSession(String sessionId) async {
    await methodChannel.invokeMethod<void>('closeSession', {'sessionId': sessionId});
  }

  @override
  Future<Map<String, dynamic>> getMetadata(String sessionId) async {
    final result = await methodChannel.invokeMethod<Map<Object?, Object?>>('getMetadata', {'sessionId': sessionId});
    return _convertMapToStringDynamic(result ?? {});
  }

  @override
  Future<List<Map<String, dynamic>>> getInputInfo(String sessionId) async {
    final result = await methodChannel.invokeMethod<List<Object?>>('getInputInfo', {'sessionId': sessionId});
    return result?.map((item) => _convertMapToStringDynamic(item as Map<Object?, Object?>)).toList() ?? [];
  }

  @override
  Future<List<Map<String, dynamic>>> getOutputInfo(String sessionId) async {
    final result = await methodChannel.invokeMethod<List<Object?>>('getOutputInfo', {'sessionId': sessionId});
    return result?.map((item) => _convertMapToStringDynamic(item as Map<Object?, Object?>)).toList() ?? [];
  }

  // OrtValue operations

  @override
  Future<Map<String, dynamic>> createOrtValue(String sourceType, dynamic data, List<int> shape) async {
    final result = await methodChannel.invokeMethod<Map<Object?, Object?>>('createOrtValue', {
      'sourceType': sourceType,
      'data': data,
      'shape': shape,
    });
    return _convertMapToStringDynamic(result ?? {});
  }

  @override
  Future<Map<String, dynamic>> convertOrtValue(String valueId, String targetType) async {
    final result = await methodChannel.invokeMethod<Map<Object?, Object?>>('convertOrtValue', {
      'valueId': valueId,
      'targetType': targetType,
    });
    return _convertMapToStringDynamic(result ?? {});
  }

  @override
  Future<Map<String, dynamic>> moveOrtValueToDevice(String valueId, String targetDevice) async {
    final result = await methodChannel.invokeMethod<Map<Object?, Object?>>('moveOrtValueToDevice', {
      'valueId': valueId,
      'targetDevice': targetDevice,
    });
    return _convertMapToStringDynamic(result ?? {});
  }

  @override
  Future<Map<String, dynamic>> getOrtValueData(String valueId, String dataType) async {
    final result = await methodChannel.invokeMethod<Map<Object?, Object?>>('getOrtValueData', {
      'valueId': valueId,
      'dataType': dataType,
    });
    return _convertMapToStringDynamic(result ?? {});
  }

  @override
  Future<void> releaseOrtValue(String valueId) async {
    await methodChannel.invokeMethod<void>('releaseOrtValue', {'valueId': valueId});
  }

  Map<String, dynamic> _convertMapToStringDynamic(Map<Object?, Object?> map) {
    return map.map((key, value) => MapEntry(key.toString(), value));
  }
}
