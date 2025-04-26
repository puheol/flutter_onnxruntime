#ifndef FLUTTER_ONNXRUNTIME_PLUGIN_H_
#define FLUTTER_ONNXRUNTIME_PLUGIN_H_

#include "export.h"

#if defined(__cplusplus)
extern "C" {
#endif

FLUTTER_PLUGIN_EXPORT void FlutterOnnxruntimePluginRegisterWithRegistrar(
    struct FlutterDesktopPluginRegistrarRef* registrar);

#if defined(__cplusplus)
}  // extern "C"
#endif

#endif  // FLUTTER_ONNXRUNTIME_PLUGIN_H_ 