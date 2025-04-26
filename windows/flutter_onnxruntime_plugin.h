#ifndef FLUTTER_PLUGIN_FLUTTER_ONNXRUNTIME_PLUGIN_H_
#define FLUTTER_PLUGIN_FLUTTER_ONNXRUNTIME_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace flutter_onnxruntime {

class FlutterOnnxruntimePlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  FlutterOnnxruntimePlugin();

  virtual ~FlutterOnnxruntimePlugin();

  // Disallow copy and assign.
  FlutterOnnxruntimePlugin(const FlutterOnnxruntimePlugin&) = delete;
  FlutterOnnxruntimePlugin& operator=(const FlutterOnnxruntimePlugin&) = delete;

  // Called when a method is called on this plugin's channel from Dart.
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace flutter_onnxruntime

#endif  // FLUTTER_PLUGIN_FLUTTER_ONNXRUNTIME_PLUGIN_H_ 