#ifndef FLUTTER_ONNXRUNTIME_TENSOR_MANAGER_H_
#define FLUTTER_ONNXRUNTIME_TENSOR_MANAGER_H_

#include "pch.h"

namespace flutter_onnxruntime {

// Manages OrtValue objects (tensors) with safe memory management
class TensorManager {
public:
    TensorManager();
    ~TensorManager();

    // Disallow copy and assign
    TensorManager(const TensorManager&) = delete;
    TensorManager& operator=(const TensorManager&) = delete;

    // Create a new tensor from Flutter data
    std::string createTensor(
        const flutter::EncodableValue& data,
        const flutter::EncodableList& shape,
        int64_t elementType);

    // Get a tensor by ID
    Ort::Value* getTensor(const std::string& tensorId);

    // Release a tensor and free resources
    bool releaseTensor(const std::string& tensorId);

    // Get data from a tensor
    flutter::EncodableValue getTensorData(
        const std::string& tensorId);

    // Convert a tensor to a different data type
    std::string convertTensor(
        const std::string& tensorId,
        int64_t newElementType);

    // Get tensor info
    flutter::EncodableMap getTensorInfo(const std::string& tensorId);

private:
    // Private implementation details
    std::unordered_map<std::string, std::unique_ptr<Ort::Value>> tensors_;
    std::mutex tensorsMutex_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Memory for tensor data that needs to persist
    std::unordered_map<std::string, std::vector<uint8_t>> tensorDataBuffers_;

    // Helper methods for tensor creation and conversion
    std::unique_ptr<Ort::Value> createOrtValue(
        const void* data,
        const std::vector<int64_t>& shape,
        ONNXTensorElementDataType elementType);

    // Extract data from a tensor
    template<typename T>
    flutter::EncodableValue extractTensorData(
        const Ort::Value* tensor,
        const std::vector<int64_t>& shape);

    // Generate a unique tensor ID
    std::string generateTensorId();
};

} // namespace flutter_onnxruntime

#endif // FLUTTER_ONNXRUNTIME_TENSOR_MANAGER_H_ 