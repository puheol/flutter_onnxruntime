// Copyright (c) MASIC AI
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "tensor_manager.h"
#include "value_conversion.h"
#include <random>

namespace flutter_onnxruntime {

TensorManager::TensorManager() : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

TensorManager::~TensorManager() {
  std::lock_guard<std::mutex> lock(tensor_mutex_);
  tensors_.clear();
  tensor_types_.clear();
  tensor_shapes_.clear();
  tensor_data_buffers_.clear();
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

std::string TensorManager::createFloat32Tensor(const std::vector<float> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();

    // Create a persistent copy of the data
    std::vector<uint8_t> data_buffer(reinterpret_cast<const uint8_t *>(data.data()),
                                     reinterpret_cast<const uint8_t *>(data.data()) + data.size() * sizeof(float));
    tensor_data_buffers_[tensor_id] = std::move(data_buffer);

    // Create a new tensor with our persistent copy of the data
    auto tensor = createOrtValue(tensor_data_buffers_[tensor_id].data(), shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::move(tensor);
    tensor_types_[tensor_id] = "float32";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &) {
    // Re-throw the exception
    throw;
  }
}

std::string TensorManager::createInt32Tensor(const std::vector<int32_t> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();

    // Create a persistent copy of the data
    std::vector<uint8_t> data_buffer(reinterpret_cast<const uint8_t *>(data.data()),
                                     reinterpret_cast<const uint8_t *>(data.data()) + data.size() * sizeof(int32_t));
    tensor_data_buffers_[tensor_id] = std::move(data_buffer);

    // Create a new tensor with our persistent copy of the data
    auto tensor = createOrtValue(tensor_data_buffers_[tensor_id].data(), shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::move(tensor);
    tensor_types_[tensor_id] = "int32";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &) {
    // Re-throw the exception
    throw;
  }
}

std::string TensorManager::createInt64Tensor(const std::vector<int64_t> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();

    // Create a persistent copy of the data
    std::vector<uint8_t> data_buffer(reinterpret_cast<const uint8_t *>(data.data()),
                                     reinterpret_cast<const uint8_t *>(data.data()) + data.size() * sizeof(int64_t));
    tensor_data_buffers_[tensor_id] = std::move(data_buffer);

    // Create a new tensor with our persistent copy of the data
    auto tensor = createOrtValue(tensor_data_buffers_[tensor_id].data(), shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::move(tensor);
    tensor_types_[tensor_id] = "int64";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &) {
    // Re-throw the exception
    throw;
  }
}

std::string TensorManager::createUint8Tensor(const std::vector<uint8_t> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();

    // Create a persistent copy of the data
    std::vector<uint8_t> data_buffer(data.begin(), data.end());
    tensor_data_buffers_[tensor_id] = std::move(data_buffer);

    // Create a new tensor with our persistent copy of the data
    auto tensor = createOrtValue(tensor_data_buffers_[tensor_id].data(), shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::move(tensor);
    tensor_types_[tensor_id] = "uint8";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &) {
    // Re-throw the exception
    throw;
  }
}

std::string TensorManager::createBoolTensor(const std::vector<bool> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();

    // Create a regular array for boolean data (std::vector<bool> is specialized and can't be used directly)
    std::vector<uint8_t> data_buffer(data.size() * sizeof(bool));
    bool *bool_data = reinterpret_cast<bool *>(data_buffer.data());
    for (size_t i = 0; i < data.size(); i++) {
      bool_data[i] = data[i];
    }
    tensor_data_buffers_[tensor_id] = std::move(data_buffer);

    // Create a new tensor with our persistent copy of the data
    auto tensor = createOrtValue(tensor_data_buffers_[tensor_id].data(), shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::move(tensor);
    tensor_types_[tensor_id] = "bool";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &) {
    // Re-throw the exception
    throw;
  }
}

flutter::EncodableValue TensorManager::getTensorData(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  // Check if the tensor exists
  auto tensor_it = tensors_.find(tensor_id);
  auto type_it = tensor_types_.find(tensor_id);
  auto shape_it = tensor_shapes_.find(tensor_id);

  if (tensor_it == tensors_.end() || type_it == tensor_types_.end() || shape_it == tensor_shapes_.end()) {
    // Return null if tensor not found
    return flutter::EncodableValue();
  }

  // Create result map
  flutter::EncodableMap result;

  try {
    // Get tensor type
    const std::string &tensor_type = type_it->second;

    // Get tensor shape
    const std::vector<int64_t> &shape = shape_it->second;

    // Convert shape to Flutter list
    flutter::EncodableList shape_list;
    for (const auto &dim : shape) {
      shape_list.push_back(static_cast<int64_t>(dim));
    }

    // Set shape and type in result
    result[flutter::EncodableValue("shape")] = flutter::EncodableValue(shape_list);
    result[flutter::EncodableValue("dataType")] = flutter::EncodableValue(tensor_type);

    // Handle different tensor types
    if (tensor_type == "float32") {
      // Get float data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      const float *tensor_data = tensor->GetTensorData<float>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<float> data_vec(tensor_data, tensor_data + elem_count);
      result[flutter::EncodableValue("data")] = ValueConversion::vectorToFlValue(data_vec);
    } else if (tensor_type == "int32") {
      // Get int32 data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      const int32_t *tensor_data = tensor->GetTensorData<int32_t>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<int32_t> data_vec(tensor_data, tensor_data + elem_count);
      result[flutter::EncodableValue("data")] = ValueConversion::vectorToFlValue(data_vec);
    } else if (tensor_type == "int64") {
      // Get int64 data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      const int64_t *tensor_data = tensor->GetTensorData<int64_t>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<int64_t> data_vec(tensor_data, tensor_data + elem_count);
      result[flutter::EncodableValue("data")] = ValueConversion::vectorToFlValue(data_vec);
    } else if (tensor_type == "uint8") {
      // Get uint8 data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      const uint8_t *tensor_data = tensor->GetTensorData<uint8_t>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<uint8_t> data_vec(tensor_data, tensor_data + elem_count);
      result[flutter::EncodableValue("data")] = ValueConversion::vectorToFlValue(data_vec);
    } else if (tensor_type == "bool") {
      // Get bool data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      const bool *tensor_data = tensor->GetTensorData<bool>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();
      std::vector<bool> data_vec(tensor_data, tensor_data + elem_count);
      result[flutter::EncodableValue("data")] = ValueConversion::vectorToFlValue(data_vec);
    } else {
      // Unsupported tensor type
      throw std::runtime_error("Unsupported tensor type: " + tensor_type);
    }
  } catch (const Ort::Exception &e) {
    throw std::runtime_error(e.what());
  }

  return flutter::EncodableValue(result);
}

bool TensorManager::releaseTensor(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  auto tensor_it = tensors_.find(tensor_id);
  auto type_it = tensor_types_.find(tensor_id);
  auto shape_it = tensor_shapes_.find(tensor_id);
  auto buffer_it = tensor_data_buffers_.find(tensor_id);

  if (tensor_it == tensors_.end()) {
    return false;
  }

  // Remove tensor, type, shape, and buffer
  tensors_.erase(tensor_it);
  if (type_it != tensor_types_.end()) {
    tensor_types_.erase(type_it);
  }
  if (shape_it != tensor_shapes_.end()) {
    tensor_shapes_.erase(shape_it);
  }
  if (buffer_it != tensor_data_buffers_.end()) {
    tensor_data_buffers_.erase(buffer_it);
  }

  return true;
}

Ort::Value *TensorManager::getTensor(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  auto it = tensors_.find(tensor_id);
  if (it == tensors_.end()) {
    return nullptr;
  }

  return it->second.get();
}

void TensorManager::storeTensor(const std::string &tensor_id, Ort::Value &&tensor) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  try {
    // Store the tensor
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));

    // Get tensor info to store type and shape
    Ort::TensorTypeAndShapeInfo tensor_info = tensors_[tensor_id]->GetTensorTypeAndShapeInfo();

    // Get and store the tensor shape
    auto shape = tensor_info.GetShape();
    tensor_shapes_[tensor_id] = shape;

    // Get and store the tensor type
    ONNXTensorElementDataType element_type = tensor_info.GetElementType();
    tensor_types_[tensor_id] = get_element_type_string(element_type);
  } catch (const std::exception &) {
    // Handle exception - just log and rethrow as needed
    throw;
  }
}

std::string TensorManager::getTensorType(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  auto it = tensor_types_.find(tensor_id);
  if (it == tensor_types_.end()) {
    throw std::runtime_error("Tensor not found");
  }

  return it->second;
}

std::vector<int64_t> TensorManager::getTensorShape(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(tensor_mutex_);

  auto it = tensor_shapes_.find(tensor_id);
  if (it == tensor_shapes_.end()) {
    throw std::runtime_error("Tensor not found");
  }

  return it->second;
}

const char *TensorManager::get_element_type_string(ONNXTensorElementDataType element_type) {
  switch (element_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "float32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return "uint8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return "int8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return "uint16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return "int16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return "int32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return "int64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    return "string";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return "bool";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return "float16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return "double";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return "uint32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return "uint64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    return "complex64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    return "complex128";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return "bfloat16";
  default:
    return "unknown";
  }
}

std::string TensorManager::convertTensor(const std::string &tensor_id, const std::string &target_type) {
  // PLACEHOLDER IMPLEMENTATION - will be implemented in the next iteration
  // For now, just return the original tensor ID to avoid breaking functionality

  // This is just a stub for now
  return tensor_id;
}

std::unique_ptr<Ort::Value> TensorManager::createOrtValue(const void *data, const std::vector<int64_t> &shape,
                                                          ONNXTensorElementDataType element_type) {
  // Calculate total element count
  size_t element_count =
      shape.empty() ? 0
                    : std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());

  // Create OrtValue based on element type
  switch (element_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    const float *tensor_data = static_cast<const float *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory_info_, const_cast<float *>(tensor_data),
                                                                        element_count, shape.data(), shape.size()));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
    const int32_t *tensor_data = static_cast<const int32_t *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<int32_t>(
        memory_info_, const_cast<int32_t *>(tensor_data), element_count, shape.data(), shape.size()));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
    const int64_t *tensor_data = static_cast<const int64_t *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<int64_t>(
        memory_info_, const_cast<int64_t *>(tensor_data), element_count, shape.data(), shape.size()));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
    const uint8_t *tensor_data = static_cast<const uint8_t *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<uint8_t>(
        memory_info_, const_cast<uint8_t *>(tensor_data), element_count, shape.data(), shape.size()));
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
    const bool *tensor_data = static_cast<const bool *>(data);
    return std::make_unique<Ort::Value>(Ort::Value::CreateTensor<bool>(memory_info_, const_cast<bool *>(tensor_data),
                                                                       element_count, shape.data(), shape.size()));
  }
  default:
    throw std::runtime_error("Unsupported tensor element type");
  }
}

} // namespace flutter_onnxruntime