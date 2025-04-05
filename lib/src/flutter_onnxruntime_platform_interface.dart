import 'package:flutter_onnxruntime/src/flutter_onnxruntime_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

abstract class FlutterOnnxruntimePlatform extends PlatformInterface {
  /// Constructs a FlutterOnnxruntimePlatform.
  FlutterOnnxruntimePlatform() : super(token: _token);

  static final Object _token = Object();

  static FlutterOnnxruntimePlatform _instance = MethodChannelFlutterOnnxruntime();

  /// The default instance of [FlutterOnnxruntimePlatform] to use.
  ///
  /// Defaults to [MethodChannelFlutterOnnxruntime].
  static FlutterOnnxruntimePlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [FlutterOnnxruntimePlatform] when
  /// they register themselves.
  static set instance(FlutterOnnxruntimePlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }

  // Core ONNX Runtime operations
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) {
    throw UnimplementedError('createSession() has not been implemented.');
  }

  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, dynamic> inputs, {
    Map<String, dynamic>? runOptions,
  }) {
    throw UnimplementedError('runInference() has not been implemented.');
  }

  Future<void> closeSession(String sessionId) {
    throw UnimplementedError('closeSession() has not been implemented.');
  }
}
