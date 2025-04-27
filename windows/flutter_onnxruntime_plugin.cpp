// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "flutter_onnxruntime_plugin.h"

// This must be included before many other Windows headers.
#include <windows.h>

// For getPlatformVersion; remove unless needed for your plugin implementation.
#include <VersionHelpers.h>

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <memory>
#include <sstream>

// Include our implementation headers
#include "src/session_manager.h"
#include "src/tensor_manager.h"
#include "src/value_conversion.h"
#include "src/windows_utils.h"

#include "include/flutter_onnxruntime/export.h"

namespace flutter_onnxruntime {

// Private implementation class to hold managers
class FlutterOnnxruntimePluginImpl {
public:
  FlutterOnnxruntimePluginImpl()
      : sessionManager_(std::make_unique<SessionManager>()), tensorManager_(std::make_unique<TensorManager>()) {}

  // Manager instances
  std::unique_ptr<SessionManager> sessionManager_;
  std::unique_ptr<TensorManager> tensorManager_;
};

// static
void FlutterOnnxruntimePlugin::RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar) {
  auto channel = std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
      registrar->messenger(), "flutter_onnxruntime", &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<FlutterOnnxruntimePlugin>();

  channel->SetMethodCallHandler([plugin_pointer = plugin.get()](const auto &call, auto result) {
    plugin_pointer->HandleMethodCall(call, std::move(result));
  });

  registrar->AddPlugin(std::move(plugin));
}

FlutterOnnxruntimePlugin::FlutterOnnxruntimePlugin() : impl_(std::make_unique<FlutterOnnxruntimePluginImpl>()) {}

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

  // OrtValue-related methods
  if (method_name == "createOrtValue") {
    HandleCreateOrtValue(method_call, std::move(result));
    return;
  } else if (method_name == "convertOrtValue") {
    HandleConvertOrtValue(method_call, std::move(result));
    return;
  } else if (method_name == "getOrtValueData") {
    HandleGetOrtValueData(method_call, std::move(result));
    return;
  } else if (method_name == "releaseOrtValue") {
    HandleReleaseOrtValue(method_call, std::move(result));
    return;
  }

  // Session-related methods (placeholders for now)
  if (method_name == "createSession" || method_name == "runInference" || method_name == "closeSession" ||
      method_name == "getMetadata" || method_name == "getInputInfo" || method_name == "getOutputInfo" ||
      method_name == "getAvailableProviders") {
    // Return placeholder responses for now
    flutter::EncodableMap placeholder;
    placeholder[flutter::EncodableValue("status")] = flutter::EncodableValue("placeholder_implementation");
    placeholder[flutter::EncodableValue("message")] =
        flutter::EncodableValue("This functionality will be implemented in the next iteration");
    result->Success(flutter::EncodableValue(placeholder));
    return;
  }

  result->NotImplemented();
}

void FlutterOnnxruntimePlugin::HandleCreateOrtValue(
    const flutter::MethodCall<flutter::EncodableValue> &method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {

  // Extract parameters
  const auto *args = std::get_if<flutter::EncodableMap>(method_call.arguments());

  if (!args) {
    result->Error("INVALID_ARG", "Arguments must be provided as a map", nullptr);
    return;
  }

  try {
    // Extract source type
    auto source_type_it = args->find(flutter::EncodableValue("sourceType"));
    if (source_type_it == args->end() || !std::holds_alternative<std::string>(source_type_it->second)) {
      result->Error("INVALID_ARG", "Source type must be a non-null string", nullptr);
      return;
    }
    std::string source_type = std::get<std::string>(source_type_it->second);

    // Extract data
    auto data_it = args->find(flutter::EncodableValue("data"));
    if (data_it == args->end()) {
      result->Error("INVALID_ARG", "Data must be provided", nullptr);
      return;
    }
    const flutter::EncodableValue &data = data_it->second;

    // Extract shape
    auto shape_it = args->find(flutter::EncodableValue("shape"));
    if (shape_it == args->end() || !std::holds_alternative<flutter::EncodableList>(shape_it->second)) {
      result->Error("INVALID_ARG", "Shape must be a non-null list", nullptr);
      return;
    }
    const flutter::EncodableList &shape = std::get<flutter::EncodableList>(shape_it->second);

    // Determine ONNX element type from source type
    ONNXTensorElementDataType element_type = ValueConversion::stringToElementType(source_type);

    // Create the tensor
    std::string tensor_id = impl_->tensorManager_->createTensor(data, shape, static_cast<int64_t>(element_type));

    // Return success with the tensor ID
    flutter::EncodableMap response;
    response[flutter::EncodableValue("valueId")] = flutter::EncodableValue(tensor_id);
    response[flutter::EncodableValue("dataType")] = flutter::EncodableValue(source_type);
    response[flutter::EncodableValue("shape")] = flutter::EncodableValue(shape);

    result->Success(flutter::EncodableValue(response));
  } catch (const Ort::Exception &e) {
    result->Error("ORT_ERROR", e.what(), nullptr);
  } catch (const std::exception &e) {
    result->Error("PLUGIN_ERROR", e.what(), nullptr);
  } catch (...) {
    result->Error("INTERNAL_ERROR", "Unknown error occurred", nullptr);
  }
}

void FlutterOnnxruntimePlugin::HandleConvertOrtValue(
    const flutter::MethodCall<flutter::EncodableValue> &method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {

  // Extract parameters
  const auto *args = std::get_if<flutter::EncodableMap>(method_call.arguments());

  if (!args) {
    result->Error("INVALID_ARG", "Arguments must be provided as a map", nullptr);
    return;
  }

  try {
    // Extract value ID
    auto value_id_it = args->find(flutter::EncodableValue("valueId"));
    if (value_id_it == args->end() || !std::holds_alternative<std::string>(value_id_it->second)) {
      result->Error("INVALID_ARG", "Value ID must be a non-null string", nullptr);
      return;
    }
    std::string value_id = std::get<std::string>(value_id_it->second);

    // Extract target type
    auto target_type_it = args->find(flutter::EncodableValue("targetType"));
    if (target_type_it == args->end() || !std::holds_alternative<std::string>(target_type_it->second)) {
      result->Error("INVALID_ARG", "Target type must be a non-null string", nullptr);
      return;
    }
    std::string target_type = std::get<std::string>(target_type_it->second);

    // Determine ONNX element type from target type
    ONNXTensorElementDataType element_type = ValueConversion::stringToElementType(target_type);

    // Convert the tensor
    std::string new_tensor_id = impl_->tensorManager_->convertTensor(value_id, static_cast<int64_t>(element_type));

    // Return success with the new tensor ID
    flutter::EncodableMap response;
    response[flutter::EncodableValue("valueId")] = flutter::EncodableValue(new_tensor_id);
    response[flutter::EncodableValue("dataType")] = flutter::EncodableValue(target_type);

    result->Success(flutter::EncodableValue(response));
  } catch (const Ort::Exception &e) {
    result->Error("ORT_ERROR", e.what(), nullptr);
  } catch (const std::exception &e) {
    result->Error("PLUGIN_ERROR", e.what(), nullptr);
  } catch (...) {
    result->Error("INTERNAL_ERROR", "Unknown error occurred", nullptr);
  }
}

void FlutterOnnxruntimePlugin::HandleGetOrtValueData(
    const flutter::MethodCall<flutter::EncodableValue> &method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {

  // Extract parameters
  const auto *args = std::get_if<flutter::EncodableMap>(method_call.arguments());

  if (!args) {
    result->Error("INVALID_ARG", "Arguments must be provided as a map", nullptr);
    return;
  }

  try {
    // Extract value ID
    auto value_id_it = args->find(flutter::EncodableValue("valueId"));
    if (value_id_it == args->end() || !std::holds_alternative<std::string>(value_id_it->second)) {
      result->Error("INVALID_ARG", "Value ID must be a non-null string", nullptr);
      return;
    }
    std::string value_id = std::get<std::string>(value_id_it->second);

    // Get the tensor data
    flutter::EncodableValue tensor_data = impl_->tensorManager_->getTensorData(value_id);

    // Return success with the tensor data
    if (std::holds_alternative<flutter::EncodableMap>(tensor_data)) {
      result->Success(tensor_data);
    } else {
      result->Error("INVALID_VALUE", "Tensor not found or invalid", nullptr);
    }
  } catch (const Ort::Exception &e) {
    result->Error("ORT_ERROR", e.what(), nullptr);
  } catch (const std::exception &e) {
    result->Error("PLUGIN_ERROR", e.what(), nullptr);
  } catch (...) {
    result->Error("INTERNAL_ERROR", "Unknown error occurred", nullptr);
  }
}

void FlutterOnnxruntimePlugin::HandleReleaseOrtValue(
    const flutter::MethodCall<flutter::EncodableValue> &method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {

  // Extract parameters
  const auto *args = std::get_if<flutter::EncodableMap>(method_call.arguments());

  if (!args) {
    result->Error("INVALID_ARG", "Arguments must be provided as a map", nullptr);
    return;
  }

  try {
    // Extract value ID
    auto value_id_it = args->find(flutter::EncodableValue("valueId"));
    if (value_id_it == args->end() || !std::holds_alternative<std::string>(value_id_it->second)) {
      result->Error("INVALID_ARG", "Value ID must be a non-null string", nullptr);
      return;
    }
    std::string value_id = std::get<std::string>(value_id_it->second);

    // Release the tensor
    bool success = impl_->tensorManager_->releaseTensor(value_id);

    // Return success status
    if (success) {
      result->Success(nullptr);
    } else {
      result->Error("INVALID_VALUE", "Tensor not found", nullptr);
    }
  } catch (const Ort::Exception &e) {
    result->Error("ORT_ERROR", e.what(), nullptr);
  } catch (const std::exception &e) {
    result->Error("PLUGIN_ERROR", e.what(), nullptr);
  } catch (...) {
    result->Error("INTERNAL_ERROR", "Unknown error occurred", nullptr);
  }
}

} // namespace flutter_onnxruntime

// C-style function for plugin registration
// This must be implemented for the plugin to be loadable
extern "C" {
FLUTTER_PLUGIN_EXPORT void FlutterOnnxruntimePluginRegisterWithRegistrar(FlutterDesktopPluginRegistrarRef registrar) {
  // Convert the C-style registrar to the C++ one
  flutter::PluginRegistrarWindows *plugin_registrar =
      flutter::PluginRegistrarManager::GetInstance()->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);

  // Call our plugin's static registration method
  flutter_onnxruntime::FlutterOnnxruntimePlugin::RegisterWithRegistrar(plugin_registrar);
}
}