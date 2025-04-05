import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:flutter_onnxruntime/src/flutter_onnxruntime_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterOnnxruntimePlatform with MockPlatformInterfaceMixin implements FlutterOnnxruntimePlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');

  @override
  Future<Map<String, dynamic>> createSession(String modelPath, {Map<String, dynamic>? sessionOptions}) =>
      Future.value({});

  @override
  Future<Map<String, dynamic>> runInference(
    String sessionId,
    Map<String, dynamic> inputs, {
    Map<String, dynamic>? runOptions,
  }) => Future.value({});

  @override
  Future<void> closeSession(String sessionId) => Future.value();
}

void main() {
  final FlutterOnnxruntimePlatform initialPlatform = FlutterOnnxruntimePlatform.instance;

  test('$MethodChannelFlutterOnnxruntime is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelFlutterOnnxruntime>());
  });

  test('getPlatformVersion', () async {
    Onnxruntime flutterOnnxruntimePlugin = Onnxruntime();
    MockFlutterOnnxruntimePlatform fakePlatform = MockFlutterOnnxruntimePlatform();
    FlutterOnnxruntimePlatform.instance = fakePlatform;

    expect(await flutterOnnxruntimePlugin.getPlatformVersion(), '42');
  });
}
