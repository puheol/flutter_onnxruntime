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

    // Create the OrtValue
    auto tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float *>(data.data()), data.size(), shape.data(), shape.size()));

    // Important: RAII principles: Instead of moving tensor, make a more robust copy
    float *tensor_data = new float[data.size()];
    std::copy(data.begin(), data.end(), tensor_data);
    // Create a new tensor with our persistent copy of the data
    auto persistent_tensor =
        Ort::Value::CreateTensor<float>(memory_info_, tensor_data, data.size(), shape.data(), shape.size());
    // Store the tensor with direct ownership, its type, and shape
    // use std::make_unique to tie the OrtValue lifetime to the pointer
    tensors_[tensor_id] = std::make_unique<Ort::Value>(std::move(persistent_tensor));
    tensor_types_[tensor_id] = "float";
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
    if (tensor_type == "float") {
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