#ifndef TENSOR_MANAGER_H
#define TENSOR_MANAGER_H

#include <flutter_linux/flutter_linux.h>
#include <map>
#include <memory>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// Class to manage tensor data
class TensorManager {
public:
  TensorManager();
  ~TensorManager();

  // Create a tensor from Float32List data
  std::string createFloat32Tensor(const std::vector<float> &data, const std::vector<int64_t> &shape);

  // Get data from a tensor
  FlValue *getTensorData(const std::string &tensor_id);

  // Release a tensor
  bool releaseTensor(const std::string &tensor_id);

  // Get the OrtValue for a tensor ID
  Ort::Value *getTensor(const std::string &tensor_id);

private:
  // Generate a unique tensor ID
  std::string generateTensorId();

  // Map of tensor IDs to OrtValue objects
  std::map<std::string, std::unique_ptr<Ort::Value>> tensors_;

  // Map of tensor IDs to their data types
  std::map<std::string, std::string> tensor_types_;

  // Map of tensor IDs to their shapes
  std::map<std::string, std::vector<int64_t>> tensor_shapes_;

  // Counter for generating unique tensor IDs
  int next_tensor_id_;

  // Mutex for thread safety
  std::mutex mutex_;

  // Memory info for CPU memory
  Ort::MemoryInfo memory_info_{nullptr};
};

#endif // TENSOR_MANAGER_H