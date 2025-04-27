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
    const flutter::EncodableValue &data_value = data_it->second;

    // Extract shape
    auto shape_it = args->find(flutter::EncodableValue("shape"));
    if (shape_it == args->end() || !std::holds_alternative<flutter::EncodableList>(shape_it->second)) {
      result->Error("INVALID_ARG", "Shape must be a non-null list", nullptr);
      return;
    }

    // Convert shape to vector<int64_t>
    const flutter::EncodableList &shape_list = std::get<flutter::EncodableList>(shape_it->second);
    std::vector<int64_t> shape;
    shape.reserve(shape_list.size());
    for (const auto &dim : shape_list) {
      if (std::holds_alternative<int32_t>(dim)) {
        shape.push_back(std::get<int32_t>(dim));
      } else if (std::holds_alternative<int64_t>(dim)) {
        shape.push_back(std::get<int64_t>(dim));
      } else {
        result->Error("INVALID_ARG", "Shape dimensions must be integers", nullptr);
        return;
      }
    }

    // Create tensor based on source type
    // check if data_value is a typed list
    // Note: Typed list in Dart is EncodableValue type
    // List<T> in Dart is EncodableList type
    // Dart always pass typed list except for bool
    std::string tensor_id;
    if (source_type == "float32") {
      if (!std::holds_alternative<std::vector<float>>(data_value)) {
        result->Error("INVALID_ARG", "Float32 data must be a list", nullptr);
        return;
      }
      std::vector<float> float_data = std::get<std::vector<float>>(data_value);
      tensor_id = impl_->tensorManager_->createFloat32Tensor(float_data, shape);
    } else if (source_type == "int32") {
      if (!std::holds_alternative<std::vector<int32_t>>(data_value)) {
        result->Error("INVALID_ARG", "Int32 data must be a list", nullptr);
        return;
      }
      std::vector<int32_t> int32_data = std::get<std::vector<int32_t>>(data_value);
      tensor_id = impl_->tensorManager_->createInt32Tensor(int32_data, shape);
    } else if (source_type == "int64") {
      if (!std::holds_alternative<std::vector<int64_t>>(data_value)) {
        result->Error("INVALID_ARG", "Int64 data must be a list", nullptr);
        return;
      }
      std::vector<int64_t> int64_data = std::get<std::vector<int64_t>>(data_value);
      tensor_id = impl_->tensorManager_->createInt64Tensor(int64_data, shape);
    } else if (source_type == "uint8") {
      if (!std::holds_alternative<std::vector<uint8_t>>(data_value)) {
        result->Error("INVALID_ARG", "Uint8 data must be a list", nullptr);
        return;
      }
      std::vector<uint8_t> uint8_data = std::get<std::vector<uint8_t>>(data_value);
      tensor_id = impl_->tensorManager_->createUint8Tensor(uint8_data, shape);
    } else if (source_type == "bool") {
      // Note: for bool values, Dart always pass a List<bool>, not a typed list
      if (!std::holds_alternative<flutter::EncodableList>(data_value)) {
        result->Error("INVALID_ARG", "Bool data must be a list", nullptr);
        return;
      }
      auto bool_data_list = std::get<flutter::EncodableList>(data_value);
      std::vector<bool> bool_data;
      bool_data.reserve(bool_data_list.size());

      for (const auto &item : bool_data_list) {
        if (std::holds_alternative<bool>(item)) {
          bool_data.push_back(std::get<bool>(item));
        } else if (std::holds_alternative<int32_t>(item)) {
          bool_data.push_back(std::get<int32_t>(item) != 0);
        }
      }
      tensor_id = impl_->tensorManager_->createBoolTensor(bool_data, shape);
    } else {
      result->Error("INVALID_ARG", "Unsupported data type: " + source_type, nullptr);
      return;
    }

    // Return success with the tensor ID
    flutter::EncodableMap response;
    response[flutter::EncodableValue("valueId")] = flutter::EncodableValue(tensor_id);
    response[flutter::EncodableValue("dataType")] = flutter::EncodableValue(source_type);

    // Convert shape to Flutter list
    flutter::EncodableList response_shape;
    for (const auto &dim : shape) {
      response_shape.push_back(static_cast<int64_t>(dim));
    }
    response[flutter::EncodableValue("shape")] = flutter::EncodableValue(response_shape);

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

    // Convert the tensor
    std::string new_tensor_id = impl_->tensorManager_->convertTensor(value_id, target_type);

    // Get the tensor shape
    std::vector<int64_t> shape = impl_->tensorManager_->getTensorShape(new_tensor_id);

    // Convert shape to Flutter list
    flutter::EncodableList shape_list;
    for (const auto &dim : shape) {
      shape_list.push_back(static_cast<int64_t>(dim));
    }

    // Return success with the new tensor ID
    flutter::EncodableMap response;
    response[flutter::EncodableValue("valueId")] = flutter::EncodableValue(new_tensor_id);
    response[flutter::EncodableValue("dataType")] = flutter::EncodableValue(target_type);
    response[flutter::EncodableValue("shape")] = flutter::EncodableValue(shape_list);

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
    result->Success(tensor_data);
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