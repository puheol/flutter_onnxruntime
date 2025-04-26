#ifndef FLUTTER_ONNXRUNTIME_VALUE_CONVERSION_H_
#define FLUTTER_ONNXRUNTIME_VALUE_CONVERSION_H_

#include "pch.h"

namespace flutter_onnxruntime {

// Utility class for converting between Flutter values and C++ types
class ValueConversion {
public:
    // Convert Flutter list to C++ vector of specified type
    template<typename T>
    static std::vector<T> flValueToVector(const flutter::EncodableList& list);

    // Convert C++ vector to Flutter list
    template<typename T>
    static flutter::EncodableList vectorToFlValue(const std::vector<T>& vec);

    // Extract tensor data from Flutter value
    static std::pair<std::vector<uint8_t>, size_t> flValueToTensorData(
        const flutter::EncodableValue& value,
        ONNXTensorElementDataType elementType);

    // Convert tensor data to Flutter value
    static flutter::EncodableValue tensorDataToFlValue(
        const void* data,
        size_t elementCount,
        ONNXTensorElementDataType elementType);

    // Get element size in bytes for ONNX type
    static size_t getElementSize(ONNXTensorElementDataType elementType);

    // Convert element type enum to string
    static std::string elementTypeToString(ONNXTensorElementDataType elementType);

    // Convert string to element type enum
    static ONNXTensorElementDataType stringToElementType(const std::string& typeStr);
};

} // namespace flutter_onnxruntime

#endif // FLUTTER_ONNXRUNTIME_VALUE_CONVERSION_H_ 