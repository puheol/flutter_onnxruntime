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

  Map<String, dynamic> _convertMapToStringDynamic(Map<Object?, Object?> map) {
    return map.map((key, value) => MapEntry(key.toString(), value));
  }
}
