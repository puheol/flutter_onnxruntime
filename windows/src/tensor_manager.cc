// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "tensor_manager.h"
#include "value_conversion.h"
#include <random>

namespace flutter_onnxruntime {

TensorManager::TensorManager() {
  // Initialize ONNX Runtime memory info for CPU
  // Using arena allocator for better performance with CPU tensors
}

TensorManager::~TensorManager() {
  // Clean up tensors with a lock to ensure thread safety
  std::lock_guard<std::mutex> lock(tensorsMutex_);
  tensors_.clear();
  tensorDataBuffers_.clear();
}

std::string TensorManager::createTensor(const flutter::EncodableValue &data, const flutter::EncodableList &shapeList,
                                        int64_t elementType) {

  // Convert shape list to vector<int64_t>
  std::vector<int64_t> shape;
  shape.reserve(shapeList.size());
  for (const auto &dim : shapeList) {
    if (std::holds_alternative<int32_t>(dim)) {
      shape.push_back(std::get<int32_t>(dim));
    } else if (std::holds_alternative<int64_t>(dim)) {
      shape.push_back(std::get<int64_t>(dim));
    } else {
      // Invalid shape dimension
      throw std::runtime_error("Invalid shape dimension type");
    }
  }

  // Convert to ONNX tensor element type
  ONNXTensorElementDataType ortElementType = static_cast<ONNXTensorElementDataType>(elementType);

  // Extract data from Flutter value
  auto [dataBuffer, elementCount] = ValueConversion::flValueToTensorData(data, ortElementType);

  // Validate data size against shape
  size_t expectedElements = 1;
  for (const auto &dim : shape) {
    expectedElements *= static_cast<size_t>(dim);
  }

  if (elementCount != expectedElements) {
    throw std::runtime_error("Data size does not match shape dimensions");
  }

  // Create OrtValue
  std::lock_guard<std::mutex> lock(tensorsMutex_);
  std::string tensorId = generateTensorId();

  try {
    // Store the data buffer so it persists
    tensorDataBuffers_[tensorId] = std::move(dataBuffer);

    // Create OrtValue
    auto ortValue = createOrtValue(tensorDataBuffers_[tensorId].data(), shape, ortElementType);

    // Store the OrtValue with ownership
    tensors_[tensorId] = std::move(ortValue);

    return tensorId;
  } catch (...) {
    // Clean up any allocated resources on failure
    tensorDataBuffers_.erase(tensorId);
    throw;
  }
}

Ort::Value *TensorManager::getTensor(const std::string &tensorId) {
  std::lock_guard<std::mutex> lock(tensorsMutex_);
  auto it = tensors_.find(tensorId);
  if (it == tensors_.end()) {
    return nullptr;
  }
  return it->second.get();
}

bool TensorManager::releaseTensor(const std::string &tensorId) {
  std::lock_guard<std::mutex> lock(tensorsMutex_);

  // Erase both the tensor and its associated data buffer if they exist
  bool found = tensors_.erase(tensorId) > 0;
  tensorDataBuffers_.erase(tensorId);

  return found;
}

flutter::EncodableValue TensorManager::getTensorData(const std::string &tensorId) {
  std::lock_guard<std::mutex> lock(tensorsMutex_);

  auto it = tensors_.find(tensorId);
  if (it == tensors_.end()) {
    // Return null if tensor not found
    return flutter::EncodableValue();
  }

  Ort::Value *tensor = it->second.get();

  try {
    // Get tensor type and shape info
    Ort::TensorTypeAndShapeInfo info = tensor->GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType elementType = info.GetElementType();
    std::vector<int64_t> shape = info.GetShape();
    size_t elementCount = info.GetElementCount();

    // Get tensor data
    const void *data = nullptr;
    switch (elementType) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      data = tensor->GetTensorData<float>();
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      data = tensor->GetTensorData<int32_t>();
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      data = tensor->GetTensorData<int64_t>();
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      data = tensor->GetTensorData<uint8_t>();
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      data = tensor->GetTensorData<bool>();
      break;
    default:
      throw std::runtime_error("Unsupported tensor element type");
    }

    // Convert tensor data to Flutter value
    flutter::EncodableValue dataValue = ValueConversion::tensorDataToFlValue(data, elementCount, elementType);

    // Convert shape to Flutter list
    flutter::EncodableList shapeList;
    for (const auto &dim : shape) {
      shapeList.push_back(static_cast<int64_t>(dim));
    }

    // Create result map
    flutter::EncodableMap result;
    result[flutter::EncodableValue("data")] = dataValue;
    result[flutter::EncodableValue("shape")] = flutter::EncodableValue(shapeList);
    result[flutter::EncodableValue("dataType")] =
        flutter::EncodableValue(ValueConversion::elementTypeToString(elementType));

    return flutter::EncodableValue(result);
  } catch (...) {
    throw;
  }
}

std::string TensorManager::convertTensor(const std::string &tensorId, int64_t newElementType) {

  std::lock_guard<std::mutex> lock(tensorsMutex_);

  auto it = tensors_.find(tensorId);
  if (it == tensors_.end()) {
    throw std::runtime_error("Tensor not found");
  }

  Ort::Value *tensor = it->second.get();
  ONNXTensorElementDataType targetElementType = static_cast<ONNXTensorElementDataType>(newElementType);

  try {
    // Get tensor info
    Ort::TensorTypeAndShapeInfo info = tensor->GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType sourceElementType = info.GetElementType();
    std::vector<int64_t> shape = info.GetShape();
    size_t elementCount = info.GetElementCount();

    // If source and target types are the same, just return the original tensor ID
    if (sourceElementType == targetElementType) {
      return tensorId;
    }

    // Create a new tensor ID
    std::string newTensorId = generateTensorId();

    // Handle different conversion cases
    // This is a simplified implementation - a complete solution would handle all type conversions
    if (sourceElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        targetElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {

      // Float to Int32 conversion
      const float *sourceData = tensor->GetTensorData<float>();
      std::vector<int32_t> targetData(elementCount);

      for (size_t i = 0; i < elementCount; i++) {
        targetData[i] = static_cast<int32_t>(sourceData[i]);
      }

      // Store the data buffer
      std::vector<uint8_t> dataBuffer(targetData.size() * sizeof(int32_t));
      memcpy(dataBuffer.data(), targetData.data(), dataBuffer.size());
      tensorDataBuffers_[newTensorId] = std::move(dataBuffer);

      // Create new OrtValue
      tensors_[newTensorId] = createOrtValue(tensorDataBuffers_[newTensorId].data(), shape, targetElementType);

      return newTensorId;
    } else if (sourceElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
               targetElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {

      // Int32 to Float conversion
      const int32_t *sourceData = tensor->GetTensorData<int32_t>();
      std::vector<float> targetData(elementCount);

      for (size_t i = 0; i < elementCount; i++) {
        targetData[i] = static_cast<float>(sourceData[i]);
      }

      // Store the data buffer
      std::vector<uint8_t> dataBuffer(targetData.size() * sizeof(float));
      memcpy(dataBuffer.data(), targetData.data(), dataBuffer.size());
      tensorDataBuffers_[newTensorId] = std::move(dataBuffer);

      // Create new OrtValue
      tensors_[newTensorId] = createOrtValue(tensorDataBuffers_[newTensorId].data(), shape, targetElementType);

      return newTensorId;
    }
    // Add more conversion cases as needed
    else {
      throw std::runtime_error("Unsupported tensor conversion");
    }
  } catch (...) {
    throw;
  }
}

flutter::EncodableMap TensorManager::getTensorInfo(const std::string &tensorId) {
  std::lock_guard<std::mutex> lock(tensorsMutex_);

  auto it = tensors_.find(tensorId);
  if (it == tensors_.end()) {
    throw std::runtime_error("Tensor not found");
  }

  Ort::Value *tensor = it->second.get();

  try {
    // Get tensor info
    Ort::TensorTypeAndShapeInfo info = tensor->GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType elementType = info.GetElementType();
    std::vector<int64_t> shape = info.GetShape();
    size_t elementCount = info.GetElementCount();

    // Convert shape to Flutter list
    flutter::EncodableList shapeList;
    for (const auto &dim : shape) {
      shapeList.push_back(static_cast<int64_t>(dim));
    }

    // Create result map
    flutter::EncodableMap result;
    result[flutter::EncodableValue("shape")] = flutter::EncodableValue(shapeList);
    result[flutter::EncodableValue("dataType")] =
        flutter::EncodableValue(ValueConversion::elementTypeToString(elementType));
    result[flutter::EncodableValue("elementCount")] = flutter::EncodableValue(static_cast<int64_t>(elementCount));

    return result;
  } catch (...) {
    throw;
  }
}

std::unique_ptr<Ort::Value> TensorManager::createOrtValue(const void *data, const std::vector<int64_t> &shape,
                                                          ONNXTensorElementDataType elementType) {

  // Create memory info for CPU
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Calculate total element count
  size_t elementCount =
      shape.empty() ? 0
                    : std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());

  // Create OrtValue based on element type
  switch (elementType) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    // Use the raw pointer directly if the data is guaranteed to remain valid
    const float *tensor_data = static_cast<const float *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float *>(tensor_data),
                                                                        elementCount, shape.data(), shape.size()));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
    const int32_t *tensor_data = static_cast<const int32_t *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<int32_t>(
        memoryInfo, const_cast<int32_t *>(tensor_data), elementCount, shape.data(), shape.size()));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
    // Use the raw pointer directly if the data is guaranteed to remain valid
    const int64_t *tensor_data = static_cast<const int64_t *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, const_cast<int64_t *>(tensor_data), elementCount, shape.data(), shape.size()));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
    // Use the raw pointer directly if the data is guaranteed to remain valid
    const uint8_t *tensor_data = static_cast<const uint8_t *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<uint8_t>(
        memoryInfo, const_cast<uint8_t *>(tensor_data), elementCount, shape.data(), shape.size()));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
    // Use the raw pointer directly if the data is guaranteed to remain valid
    const bool *tensor_data = static_cast<const bool *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<bool>(memoryInfo, const_cast<bool *>(tensor_data),
                                                                       elementCount, shape.data(), shape.size()));
  }
  default:
    throw std::runtime_error("Unsupported tensor element type");
  }
}

std::string TensorManager::generateTensorId() {
  // Create a random tensor ID
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);

  std::stringstream ss;
  ss << "tensor_";
  for (int i = 0; i < 16; i++) {
    ss << std::hex << dis(gen);
  }

  return ss.str();
}

} // namespace flutter_onnxruntime