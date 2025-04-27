// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "value_conversion.h"

namespace flutter_onnxruntime {

// Helper function to get element type size
size_t ValueConversion::getElementSize(ONNXTensorElementDataType elementType) {
  switch (elementType) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return sizeof(float);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return sizeof(int32_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return sizeof(int64_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return sizeof(uint8_t);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return sizeof(bool);
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    // Variable size, handle differently
    return 0;
  default:
    // Unsupported type
    return 0;
  }
}

// Convert element type enum to string
std::string ValueConversion::elementTypeToString(ONNXTensorElementDataType elementType) {
  switch (elementType) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "float32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return "int32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return "int64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return "uint8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return "bool";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    return "string";
  default:
    return "unknown";
  }
}

// Convert string to element type enum
ONNXTensorElementDataType ValueConversion::stringToElementType(const std::string &typeStr) {
  if (typeStr == "float32") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (typeStr == "int32") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  } else if (typeStr == "int64") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  } else if (typeStr == "uint8") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  } else if (typeStr == "bool") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  } else if (typeStr == "string") {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  } else {
    // Default to float if unknown
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
}

// Specialization for float vectors
template <> std::vector<float> ValueConversion::flValueToVector<float>(const flutter::EncodableList &list) {
  std::vector<float> result;
  result.reserve(list.size());

  for (const auto &item : list) {
    if (std::holds_alternative<double>(item)) {
      result.push_back(static_cast<float>(std::get<double>(item)));
    }
  }

  return result;
}

// Specialization for int32_t vectors
template <> std::vector<int32_t> ValueConversion::flValueToVector<int32_t>(const flutter::EncodableList &list) {
  std::vector<int32_t> result;
  result.reserve(list.size());

  for (const auto &item : list) {
    if (std::holds_alternative<int32_t>(item)) {
      result.push_back(std::get<int32_t>(item));
    } else if (std::holds_alternative<int64_t>(item)) {
      result.push_back(static_cast<int32_t>(std::get<int64_t>(item)));
    }
  }

  return result;
}

// Specialization for int64_t vectors
template <> std::vector<int64_t> ValueConversion::flValueToVector<int64_t>(const flutter::EncodableList &list) {
  std::vector<int64_t> result;
  result.reserve(list.size());

  for (const auto &item : list) {
    if (std::holds_alternative<int32_t>(item)) {
      result.push_back(std::get<int32_t>(item));
    } else if (std::holds_alternative<int64_t>(item)) {
      result.push_back(std::get<int64_t>(item));
    }
  }

  return result;
}

// Specialization for uint8_t vectors
template <> std::vector<uint8_t> ValueConversion::flValueToVector<uint8_t>(const flutter::EncodableList &list) {
  std::vector<uint8_t> result;
  result.reserve(list.size());

  for (const auto &item : list) {
    if (std::holds_alternative<int32_t>(item)) {
      result.push_back(static_cast<uint8_t>(std::get<int32_t>(item)));
    } else if (std::holds_alternative<int64_t>(item)) {
      result.push_back(static_cast<uint8_t>(std::get<int64_t>(item)));
    }
  }

  return result;
}

// Specialization for bool vectors
template <> std::vector<bool> ValueConversion::flValueToVector<bool>(const flutter::EncodableList &list) {
  std::vector<bool> result;
  result.reserve(list.size());

  for (const auto &item : list) {
    if (std::holds_alternative<bool>(item)) {
      result.push_back(std::get<bool>(item));
    } else if (std::holds_alternative<int32_t>(item)) {
      result.push_back(std::get<int32_t>(item) != 0);
    }
  }

  return result;
}

// Specialization for float to Flutter EncodableList
template <> flutter::EncodableList ValueConversion::vectorToFlValue<float>(const std::vector<float> &vec) {
  flutter::EncodableList result;
  result.reserve(vec.size());

  for (const auto &val : vec) {
    result.push_back(static_cast<double>(val));
  }

  return result;
}

// Specialization for int32_t to Flutter EncodableList
template <> flutter::EncodableList ValueConversion::vectorToFlValue<int32_t>(const std::vector<int32_t> &vec) {
  flutter::EncodableList result;
  result.reserve(vec.size());

  for (const auto &val : vec) {
    result.push_back(val);
  }

  return result;
}

// Specialization for int64_t to Flutter EncodableList
template <> flutter::EncodableList ValueConversion::vectorToFlValue<int64_t>(const std::vector<int64_t> &vec) {
  flutter::EncodableList result;
  result.reserve(vec.size());

  for (const auto &val : vec) {
    result.push_back(val);
  }

  return result;
}

// Specialization for uint8_t to Flutter EncodableList
template <> flutter::EncodableList ValueConversion::vectorToFlValue<uint8_t>(const std::vector<uint8_t> &vec) {
  flutter::EncodableList result;
  result.reserve(vec.size());

  for (const auto &val : vec) {
    result.push_back(static_cast<int32_t>(val));
  }

  return result;
}

// Specialization for bool to Flutter EncodableList
template <> flutter::EncodableList ValueConversion::vectorToFlValue<bool>(const std::vector<bool> &vec) {
  flutter::EncodableList result;
  result.reserve(vec.size());

  for (size_t i = 0; i < vec.size(); i++) {
    result.push_back(vec[i]);
  }

  return result;
}

// Extract tensor data from Flutter value
std::pair<std::vector<uint8_t>, size_t> ValueConversion::flValueToTensorData(const flutter::EncodableValue &value,
                                                                             ONNXTensorElementDataType elementType) {

  std::vector<uint8_t> buffer;
  size_t elementCount = 0;

  if (std::holds_alternative<flutter::EncodableList>(value)) {
    const auto &list = std::get<flutter::EncodableList>(value);
    elementCount = list.size();

    switch (elementType) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      auto typedData = flValueToVector<float>(list);
      buffer.resize(typedData.size() * sizeof(float));
      memcpy(buffer.data(), typedData.data(), buffer.size());
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      auto typedData = flValueToVector<int32_t>(list);
      buffer.resize(typedData.size() * sizeof(int32_t));
      memcpy(buffer.data(), typedData.data(), buffer.size());
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      auto typedData = flValueToVector<int64_t>(list);
      buffer.resize(typedData.size() * sizeof(int64_t));
      memcpy(buffer.data(), typedData.data(), buffer.size());
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      auto typedData = flValueToVector<uint8_t>(list);
      buffer.resize(typedData.size() * sizeof(uint8_t));
      memcpy(buffer.data(), typedData.data(), buffer.size());
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
      auto typedData = flValueToVector<bool>(list);
      buffer.resize(typedData.size() * sizeof(bool));
      for (size_t i = 0; i < typedData.size(); i++) {
        static_cast<bool *>(static_cast<void *>(buffer.data()))[i] = typedData[i];
      }
      break;
    }
    default:
      // Unsupported type
      elementCount = 0;
      break;
    }
  }

  return {buffer, elementCount};
}

// Convert tensor data to Flutter value
flutter::EncodableValue ValueConversion::tensorDataToFlValue(const void *data, size_t elementCount,
                                                             ONNXTensorElementDataType elementType) {

  switch (elementType) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    const float *typedData = static_cast<const float *>(data);
    std::vector<float> vec(typedData, typedData + elementCount);
    return flutter::EncodableValue(vectorToFlValue(vec));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
    const int32_t *typedData = static_cast<const int32_t *>(data);
    std::vector<int32_t> vec(typedData, typedData + elementCount);
    return flutter::EncodableValue(vectorToFlValue(vec));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
    const int64_t *typedData = static_cast<const int64_t *>(data);
    std::vector<int64_t> vec(typedData, typedData + elementCount);
    return flutter::EncodableValue(vectorToFlValue(vec));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
    const uint8_t *typedData = static_cast<const uint8_t *>(data);
    std::vector<uint8_t> vec(typedData, typedData + elementCount);
    return flutter::EncodableValue(vectorToFlValue(vec));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
    const bool *typedData = static_cast<const bool *>(data);
    std::vector<bool> vec;
    vec.reserve(elementCount);
    for (size_t i = 0; i < elementCount; i++) {
      vec.push_back(typedData[i]);
    }
    return flutter::EncodableValue(vectorToFlValue(vec));
  }
  default:
    // Unsupported type, return empty list
    return flutter::EncodableValue(flutter::EncodableList());
  }
}

} // namespace flutter_onnxruntime