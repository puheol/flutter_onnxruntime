import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';
import 'package:flutter_onnxruntime/src/ort_model_metadata.dart';
import 'package:flutter_onnxruntime/src/ort_value.dart';

class OrtSession {
  final String id;
  final List<String> inputNames;
  final List<String> outputNames;

  // Private constructor
  OrtSession._({required this.id, required this.inputNames, required this.outputNames});

  // Public factory constructor to create from map
  factory OrtSession.fromMap(Map<String, dynamic> map) {
    return OrtSession._(
      id: map['sessionId'] as String,
      inputNames: List<String>.from(map['inputNames'] ?? []),
      outputNames: List<String>.from(map['outputNames'] ?? []),
    );
  }

  /// Run inference on the session
  ///
  /// [inputs] is a map of input names to input values
  /// [options] is an optional map of run options
  ///
  /// `inputs` can contain either raw data (lists, arrays) or OrtValue objects:
  ///
  /// Example with raw data:
  /// ```dart
  /// final inputs = {
  ///   'input_name': [[1, 2], [3, 4]],
  ///   'input_name_shape': [2, 2],
  /// };
  /// ```
  ///
  /// Example with OrtValue objects:
  /// ```dart
  /// final inputTensor = await OrtValue.fromFloat32List(
  ///   Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
  ///   [2, 2]
  /// );
  /// final inputs = {
  ///   'input_name': inputTensor,
  /// };
  /// ```
  Future<Map<String, dynamic>> run(Map<String, dynamic> inputs, {OrtRunOptions? options}) async {
    // Process inputs to handle OrtValue objects
    final processedInputs = Map<String, dynamic>.from(inputs);

    // Convert any OrtValue objects to their value IDs for platform channel transmission
    for (final entry in inputs.entries) {
      if (entry.value is OrtValue) {
        final ortValue = entry.value as OrtValue;
        // Replace OrtValue with its valueId for the platform channel
        processedInputs[entry.key] = {'valueId': ortValue.id};
        // Also include shape information
        processedInputs['${entry.key}_shape'] = ortValue.shape;
      }
    }

    return await FlutterOnnxruntimePlatform.instance.runInference(
      id,
      processedInputs,
      runOptions: options?.toMap() ?? {},
    );
  }

  Future<void> close() async {
    await FlutterOnnxruntimePlatform.instance.closeSession(id);
  }

  /// Get metadata about the model
  ///
  /// Returns information about the model such as producer name, graph name,
  /// domain, description, version, and custom metadata.
  Future<OrtModelMetadata> getMetadata() async {
    final metadataMap = await FlutterOnnxruntimePlatform.instance.getMetadata(id);
    return OrtModelMetadata.fromMap(metadataMap);
  }

  /// Get input info about the model
  ///
  /// Returns information about the model's inputs such as name, type, and shape.
  Future<List<Map<String, dynamic>>> getInputInfo() async {
    final inputInfoMap = await FlutterOnnxruntimePlatform.instance.getInputInfo(id);
    return inputInfoMap.map((info) => Map<String, dynamic>.from(info)).toList();
  }

  /// Get output info about the model
  ///
  /// Returns information about the model's outputs such as name, type, and shape.
  Future<List<Map<String, dynamic>>> getOutputInfo() async {
    final outputInfoMap = await FlutterOnnxruntimePlatform.instance.getOutputInfo(id);
    return outputInfoMap.map((info) => Map<String, dynamic>.from(info)).toList();
  }
}

class OrtSessionOptions {
  final int? intraOpNumThreads;
  final int? interOpNumThreads;
  final bool? enableCpuMemArena;

  OrtSessionOptions({this.intraOpNumThreads, this.interOpNumThreads, this.enableCpuMemArena});

  Map<String, dynamic> toMap() {
    return {
      if (intraOpNumThreads != null) 'intraOpNumThreads': intraOpNumThreads,
      if (interOpNumThreads != null) 'interOpNumThreads': interOpNumThreads,
      if (enableCpuMemArena != null) 'enableCpuMemArena': enableCpuMemArena,
    };
  }
}

class OrtRunOptions {
  final bool? logSeverityLevel;
  final bool? logVerbosityLevel;
  final bool? terminate;

  OrtRunOptions({this.logSeverityLevel, this.logVerbosityLevel, this.terminate});

  Map<String, dynamic> toMap() {
    return {
      if (logSeverityLevel != null) 'logSeverityLevel': logSeverityLevel,
      if (logVerbosityLevel != null) 'logVerbosityLevel': logVerbosityLevel,
      if (terminate != null) 'terminate': terminate,
    };
  }
}
