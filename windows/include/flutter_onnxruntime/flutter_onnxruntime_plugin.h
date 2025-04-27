#ifndef FLUTTER_ONNXRUNTIME_PLUGIN_H_
#define FLUTTER_ONNXRUNTIME_PLUGIN_H_

#include "export.h"
#include <flutter/plugin_registrar_windows.h>

#if defined(__cplusplus)
extern "C" {
#endif

FLUTTER_PLUGIN_EXPORT void FlutterOnnxruntimePluginRegisterWithRegistrar(FlutterDesktopPluginRegistrarRef registrar);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // FLUTTER_ONNXRUNTIME_PLUGIN_H_