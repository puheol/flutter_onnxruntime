#include "tensor_manager.h"
#include "value_conversion.h"

TensorManager::TensorManager()
    : next_tensor_id_(1), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

TensorManager::~TensorManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  tensors_.clear();
  tensor_types_.clear();
  tensor_shapes_.clear();
}

std::string TensorManager::generateTensorId() { return "tensor_" + std::to_string(next_tensor_id_++); }

std::string TensorManager::createFloat32Tensor(const std::vector<float> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // make a more robust copy, avoid the delocation of the original data
    float *tensor_data = new float[data.size()];
    std::copy(data.begin(), data.end(), tensor_data);
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<float>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    // Following RAII principles, use std::make_unique to tie the OrtValue lifetime to the pointer
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "float32";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    // Handle exception
    return "";
  }
}

std::string TensorManager::createInt32Tensor(const std::vector<int32_t> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // Make a robust copy of the data
    int32_t *tensor_data = new int32_t[data.size()];
    std::copy(data.begin(), data.end(), tensor_data);
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<int32_t>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "int32";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    // Handle exception
    return "";
  }
}

std::string TensorManager::createInt64Tensor(const std::vector<int64_t> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // Make a robust copy of the data
    int64_t *tensor_data = new int64_t[data.size()];
    std::copy(data.begin(), data.end(), tensor_data);
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<int64_t>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "int64";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    // Handle exception
    return "";
  }
}

std::string TensorManager::createUint8Tensor(const std::vector<uint8_t> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // Make a robust copy of the data
    uint8_t *tensor_data = new uint8_t[data.size()];
    std::copy(data.begin(), data.end(), tensor_data);
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<uint8_t>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "uint8";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    // Handle exception
    return "";
  }
}

std::string TensorManager::createBoolTensor(const std::vector<bool> &data, const std::vector<int64_t> &shape) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Create a unique tensor ID
    std::string tensor_id = generateTensorId();
    // Create a regular array for the boolean data (std::vector<bool> is specialized and can't be used directly)
    bool *tensor_data = new bool[data.size()];
    for (size_t i = 0; i < data.size(); i++) {
      tensor_data[i] = data[i];
    }
    // Create a new tensor with our persistent copy of the data
    auto tensor = Ort::Value::CreateTensor<bool>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(tensor));
    tensor_types_[tensor_id] = "bool";
    tensor_shapes_[tensor_id] = shape;

    return tensor_id;
  } catch (const Ort::Exception &e) {
    // Handle exception
    return "";
  }
}

FlValue *TensorManager::getTensorData(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if the tensor exists
  auto tensor_it = tensors_.find(tensor_id);
  auto type_it = tensor_types_.find(tensor_id);
  auto shape_it = tensor_shapes_.find(tensor_id);

  if (tensor_it == tensors_.end() || type_it == tensor_types_.end() || shape_it == tensor_shapes_.end()) {
    // Tensor not found
    return fl_value_new_null();
  }

  // Create result map
  g_autoptr(FlValue) result = fl_value_new_map();

  try {
    // Get tensor type
    const std::string &tensor_type = type_it->second;

    // Get tensor shape
    const std::vector<int64_t> &shape = shape_it->second;

    // Convert shape to FlValue
    FlValue *shape_list = fl_value_new_list();
    for (const auto &dim : shape) {
      fl_value_append_take(shape_list, fl_value_new_int(static_cast<int64_t>(dim)));
    }

    // Set shape and type in result
    fl_value_set_string_take(result, "shape", shape_list);
    fl_value_set_string_take(result, "dataType", fl_value_new_string(tensor_type.c_str()));

    // Handle different tensor types
    if (tensor_type == "float32") {
      // Get float data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      float *tensor_data = tensor->GetTensorMutableData<float>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<float> data_vec(tensor_data, tensor_data + elem_count);
      FlValue *data_list = vector_to_fl_value(data_vec);

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else if (tensor_type == "int32") {
      // Get int32 data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      int32_t *tensor_data = tensor->GetTensorMutableData<int32_t>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<int32_t> data_vec(tensor_data, tensor_data + elem_count);
      // Convert to int for Flutter
      std::vector<int> int_data_vec(data_vec.begin(), data_vec.end());
      FlValue *data_list = vector_to_fl_value(int_data_vec);

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else if (tensor_type == "int64") {
      // Get int64 data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      int64_t *tensor_data = tensor->GetTensorMutableData<int64_t>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<int64_t> data_vec(tensor_data, tensor_data + elem_count);
      FlValue *data_list = fl_value_new_list();
      for (const auto &val : data_vec) {
        fl_value_append_take(data_list, fl_value_new_int(val));
      }

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else if (tensor_type == "uint8") {
      // Get uint8 data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      uint8_t *tensor_data = tensor->GetTensorMutableData<uint8_t>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values
      std::vector<uint8_t> data_vec(tensor_data, tensor_data + elem_count);
      FlValue *data_list = fl_value_new_list();
      for (const auto &val : data_vec) {
        fl_value_append_take(data_list, fl_value_new_int(val));
      }

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else if (tensor_type == "bool") {
      // Get bool data from tensor
      Ort::Value *tensor = tensor_it->second.get();
      bool *tensor_data = tensor->GetTensorMutableData<bool>();

      // Get tensor info
      Ort::TensorTypeAndShapeInfo tensor_info = tensor->GetTensorTypeAndShapeInfo();
      size_t elem_count = tensor_info.GetElementCount();

      // Create data list and copy values - convert bool to int for Flutter compatibility
      FlValue *data_list = fl_value_new_list();
      for (size_t i = 0; i < elem_count; i++) {
        fl_value_append_take(data_list, fl_value_new_int(tensor_data[i] ? 1 : 0));
      }

      // Set data in result
      fl_value_set_string_take(result, "data", data_list);
    } else {
      // Unsupported tensor type
      fl_value_set_string_take(result, "error", fl_value_new_string("Unsupported tensor type"));
    }
  } catch (const Ort::Exception &e) {
    fl_value_set_string_take(result, "error", fl_value_new_string(e.what()));
  }

  return fl_value_ref(result);
}

bool TensorManager::releaseTensor(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto tensor_it = tensors_.find(tensor_id);
  auto type_it = tensor_types_.find(tensor_id);
  auto shape_it = tensor_shapes_.find(tensor_id);

  if (tensor_it == tensors_.end()) {
    return false;
  }

  // Remove tensor, type, and shape
  tensors_.erase(tensor_it);
  if (type_it != tensor_types_.end()) {
    tensor_types_.erase(type_it);
  }
  if (shape_it != tensor_shapes_.end()) {
    tensor_shapes_.erase(shape_it);
  }

  return true;
}

Ort::Value *TensorManager::getTensor(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = tensors_.find(tensor_id);
  if (it == tensors_.end()) {
    return nullptr;
  }

  return it->second.get();
}

void TensorManager::storeTensor(const std::string &tensor_id, Ort::Value &&tensor) {
  std::lock_guard<std::mutex> lock(mutex_);

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

    // Map ONNX type to our type string
    switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      tensor_types_[tensor_id] = "float32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      tensor_types_[tensor_id] = "uint8";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      tensor_types_[tensor_id] = "int8";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      tensor_types_[tensor_id] = "uint16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      tensor_types_[tensor_id] = "int16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      tensor_types_[tensor_id] = "int32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      tensor_types_[tensor_id] = "int64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      tensor_types_[tensor_id] = "string";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      tensor_types_[tensor_id] = "bool";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      tensor_types_[tensor_id] = "float16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      tensor_types_[tensor_id] = "double";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      tensor_types_[tensor_id] = "uint32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      tensor_types_[tensor_id] = "uint64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      tensor_types_[tensor_id] = "complex64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      tensor_types_[tensor_id] = "complex128";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      tensor_types_[tensor_id] = "bfloat16";
      break;
    default:
      tensor_types_[tensor_id] = "unknown";
      break;
    }
  } catch (const std::exception &e) {
    // Handle exception - maybe log it
  }
}

std::string TensorManager::getTensorType(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return tensor_types_.at(tensor_id);
}

std::vector<int64_t> TensorManager::getTensorShape(const std::string &tensor_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return tensor_shapes_.at(tensor_id);
}
