#include "flutter_onnxruntime_plugin.h"

// This must be included before many other Windows headers.
#include <windows.h>

// For getPlatformVersion; remove unless needed for your plugin implementation.
#include <VersionHelpers.h>

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <memory>
#include <sstream>

namespace flutter_onnxruntime {

// Forward declaration for implementation details
class FlutterOnnxruntimePluginImpl;

// static
void FlutterOnnxruntimePlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows *registrar) {
  auto channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "flutter_onnxruntime",
          &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<FlutterOnnxruntimePlugin>();

  channel->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto &call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });

  registrar->AddPlugin(std::move(plugin));
}

FlutterOnnxruntimePlugin::FlutterOnnxruntimePlugin() {}

FlutterOnnxruntimePlugin::~FlutterOnnxruntimePlugin() {}

void FlutterOnnxruntimePlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue> &method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  const auto &method_name = method_call.method_name();

  if (method_name == "getPlatformVersion") {
    std::ostringstream version_stream;
    if (IsWindows10OrGreater()) {
      version_stream << "Windows 10+";
    } else if (IsWindows8OrGreater()) {
      version_stream << "Windows 8";
    } else if (IsWindows7OrGreater()) {
      version_stream << "Windows 7";
    }
    result->Success(flutter::EncodableValue(version_stream.str()));
    return;
  }

  // TODO: Implement other methods by delegating to appropriate handlers
  // Methods to implement:
  // - createSession
  // - runInference
  // - closeSession
  // - getMetadata
  // - getInputInfo
  // - getOutputInfo
  // - createOrtValue
  // - convertOrtValue
  // - getOrtValueData
  // - releaseOrtValue
  // - getAvailableExecutionProviders
  // - getMemoryInfo
  
  result->NotImplemented();
}

}  // namespace flutter_onnxruntime 