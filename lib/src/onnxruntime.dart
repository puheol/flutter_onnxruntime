import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:flutter_onnxruntime/src/ort_session.dart';

class OnnxRuntime {
  Future<String?> getPlatformVersion() {
    return FlutterOnnxruntimePlatform.instance.getPlatformVersion();
  }

  /// Create an ONNX Runtime session with the given model path
  Future<OrtSession> createSession(String modelPath, {OrtSessionOptions? options}) async {
    final result = await FlutterOnnxruntimePlatform.instance.createSession(
      modelPath,
      sessionOptions: options?.toMap() ?? {},
    );
    return OrtSession.fromMap(result);
  }

  /// Create an ONNX Runtime session from an asset model file
  ///
  /// This will extract the asset to a temporary file and use that path
  Future<OrtSession> createSessionFromAsset(String assetKey, {OrtSessionOptions? options}) async {
    // Get the temporary directory
    final directory = await getTemporaryDirectory();
    final fileName = path.basename(assetKey);
    final filePath = path.join(directory.path, fileName);

    // Check if the model already exists
    final file = File(filePath);
    if (!await file.exists()) {
      // Extract asset to file
      final data = await rootBundle.load(assetKey);
      await file.writeAsBytes(data.buffer.asUint8List());
    }

    // Create session with the file path
    return createSession(filePath, options: options);
  }

  /// Get the available providers
  Future<List<String>> getAvailableProviders() async {
    return FlutterOnnxruntimePlatform.instance.getAvailableProviders();
  }
}
