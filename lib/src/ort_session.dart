import 'package:flutter_onnxruntime/src/flutter_onnxruntime_platform_interface.dart';

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

  Future<Map<String, dynamic>> run(Map<String, dynamic> inputs, {OrtRunOptions? options}) async {
    return await FlutterOnnxruntimePlatform.instance.runInference(id, inputs, runOptions: options?.toMap() ?? {});
  }

  Future<void> close() async {
    await FlutterOnnxruntimePlatform.instance.closeSession(id);
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
