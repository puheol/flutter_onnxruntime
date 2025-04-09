#include "ort_value.h"
#include "float16_utils.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Define debug macros
#define DEBUG_LOG(msg) std::cout << "[OrtValue DEBUG] " << msg << std::endl
#define ERROR_LOG(msg) std::cerr << "[OrtValue ERROR] " << msg << std::endl

// Maintain global mappings for OrtValue objects
std::unordered_map<std::string, std::unique_ptr<Ort::Value>> g_ort_values;

// Create a new OrtValue from data
extern "C" char *ort_create_tensor(const char *source_type, const void *data, const int64_t *shape, int shape_len,
                                   const char *target_type, const char *device, char **error_out) {

  try {
    DEBUG_LOG("ort_create_tensor called");
    DEBUG_LOG("Source type: " << (source_type ? source_type : "null"));
    DEBUG_LOG("Target type: " << (target_type ? target_type : "null (using source type)"));
    DEBUG_LOG("Device: " << (device ? device : "null (using CPU)"));
    DEBUG_LOG("Shape length: " << shape_len);

    // Log the shape
    std::ostringstream shape_str;
    shape_str << "Shape: [";
    for (int i = 0; i < shape_len; i++) {
      shape_str << shape[i];
      if (i < shape_len - 1)
        shape_str << ", ";
    }
    shape_str << "]";
    DEBUG_LOG(shape_str.str());

    if (!data) {
      ERROR_LOG("Data pointer is null");
      if (error_out) {
        *error_out = strdup("Data pointer is null");
      }
      return nullptr;
    }

    // Create ORT environment and allocator if not initialized
    static Ort::Env *env = nullptr;
    static Ort::AllocatorWithDefaultOptions allocator;
    if (env == nullptr) {
      DEBUG_LOG("Initializing ORT environment");
      env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OrtValue");
    }

    // Convert shape to vector
    std::vector<int64_t> shape_vec(shape, shape + shape_len);

    // Determine element type and size
    ONNXTensorElementDataType element_type;
    size_t element_size = 0;

    if (strcmp(source_type, "float32") == 0) {
      element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      element_size = sizeof(float);
    } else if (strcmp(source_type, "int32") == 0) {
      element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      element_size = sizeof(int32_t);
    } else if (strcmp(source_type, "int64") == 0) {
      element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      element_size = sizeof(int64_t);
    } else if (strcmp(source_type, "uint8") == 0) {
      element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      element_size = sizeof(uint8_t);
    } else if (strcmp(source_type, "bool") == 0) {
      element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
      element_size = sizeof(bool);
    } else if (strcmp(source_type, "float16") == 0) {
      element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      element_size = sizeof(uint16_t); // Float16 is stored as uint16_t
    } else {
      ERROR_LOG("Unsupported source data type: " << source_type);
      if (error_out) {
        std::string error = "Unsupported source data type: " + std::string(source_type);
        *error_out = strdup(error.c_str());
      }
      return nullptr;
    }

    // Calculate total number of elements
    size_t total_elements = 1;
    for (size_t i = 0; i < shape_vec.size(); i++) {
      total_elements *= shape_vec[i];
    }
    DEBUG_LOG("Total elements: " << total_elements);
    DEBUG_LOG("Element size: " << element_size << " bytes");
    DEBUG_LOG("Total data size: " << (total_elements * element_size) << " bytes");

    // Create memory info for the target device
    DEBUG_LOG("Creating memory info for device: " << (device ? device : "cpu"));
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Total size of data in bytes
    size_t data_size = total_elements * element_size;

    // Choose appropriate conversion based on target type
    ONNXTensorElementDataType target_element_type = element_type;
    if (target_type && *target_type) {
      DEBUG_LOG("Target type specified: " << target_type);
      if (strcmp(target_type, "float32") == 0) {
        target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      } else if (strcmp(target_type, "float16") == 0) {
        target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      } else if (strcmp(target_type, "int32") == 0) {
        target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      } else if (strcmp(target_type, "int64") == 0) {
        target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      } else if (strcmp(target_type, "uint8") == 0) {
        target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      } else if (strcmp(target_type, "bool") == 0) {
        target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
      }
    }

    // Create the tensor and potentially convert it
    std::unique_ptr<Ort::Value> tensor = std::make_unique<Ort::Value>(nullptr);

    if (element_type == target_element_type) {
      DEBUG_LOG("No conversion needed, creating tensor directly with type: " << element_type);
      // No conversion needed, create the tensor directly
      void *tensor_data = malloc(data_size);
      if (!tensor_data) {
        ERROR_LOG("Failed to allocate memory for tensor data (" << data_size << " bytes)");
        if (error_out) {
          *error_out = strdup("Failed to allocate memory for tensor data");
        }
        return nullptr;
      }

      // Copy the data as-is
      DEBUG_LOG("Copying data to tensor");
      memcpy(tensor_data, data, data_size);

      // Create the tensor
      DEBUG_LOG("Creating tensor with CreateTensor API");
      *tensor = Ort::Value::CreateTensor(memory_info, tensor_data, data_size, shape_vec.data(), shape_vec.size(),
                                         element_type);
      DEBUG_LOG("Tensor created successfully");
    } else {
      DEBUG_LOG("Conversion needed from type " << element_type << " to " << target_element_type);
      // Need conversion
      if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
          target_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        DEBUG_LOG("Converting from Float32 to Float16");
        // Float32 to Float16 conversion
        const float *float_data = static_cast<const float *>(data);

        // Allocate memory for float16 data
        std::vector<uint16_t> float16_data(total_elements);

        // Convert each float32 to float16
        for (size_t i = 0; i < total_elements; i++) {
          float16_data[i] = Float16Utils::floatToFloat16(float_data[i]);
        }

        // Create tensor with float16 data
        DEBUG_LOG("Creating tensor with float16 data");
        *tensor = Ort::Value::CreateTensor(memory_info, float16_data.data(), float16_data.size() * sizeof(uint16_t),
                                           shape_vec.data(), shape_vec.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
      } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
                 target_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        DEBUG_LOG("Converting from Float16 to Float32");
        // Float16 to Float32 conversion
        const uint16_t *float16_data = static_cast<const uint16_t *>(data);

        // Allocate memory for float32 data
        std::vector<float> float32_data(total_elements);

        // Convert each float16 to float32
        for (size_t i = 0; i < total_elements; i++) {
          float32_data[i] = Float16Utils::float16ToFloat(float16_data[i]);
        }

        // Create tensor with float32 data
        DEBUG_LOG("Creating tensor with float32 data");
        *tensor = Ort::Value::CreateTensor(memory_info, float32_data.data(), float32_data.size() * sizeof(float),
                                           shape_vec.data(), shape_vec.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      } else {
        DEBUG_LOG("Unsupported conversion, creating with source type for later conversion");
        // For other conversions, fallback to creating with source type and rely on later conversion
        void *tensor_data = malloc(data_size);
        if (!tensor_data) {
          ERROR_LOG("Failed to allocate memory for tensor data (" << data_size << " bytes)");
          if (error_out) {
            *error_out = strdup("Failed to allocate memory for tensor data");
          }
          return nullptr;
        }

        // Copy the data as-is
        memcpy(tensor_data, data, data_size);

        // Create the tensor
        DEBUG_LOG("Creating tensor with original type");
        *tensor = Ort::Value::CreateTensor(memory_info, tensor_data, data_size, shape_vec.data(), shape_vec.size(),
                                           element_type);
      }
    }

    // Store the OrtValue
    std::string id = generate_ort_value_uuid();
    DEBUG_LOG("Generated OrtValue ID: " << id);
    g_ort_values[id] = std::move(tensor);
    DEBUG_LOG("Stored OrtValue in global map (count now: " << g_ort_values.size() << ")");

    // Create result JSON
    // Format: {"valueId":"uuid","dataType":"type","shape":[1,2,3],"device":"cpu"}
    std::string result = "{\"valueId\":\"" + id + "\",\"dataType\":\"";
    result += target_type ? target_type : source_type;
    result += "\",\"shape\":[";

    for (size_t i = 0; i < shape_vec.size(); i++) {
      if (i > 0)
        result += ",";
      result += std::to_string(shape_vec[i]);
    }

    result += "],\"device\":\"";
    result += device ? device : "cpu";
    result += "\"}";

    DEBUG_LOG("Returning result: " << result);
    return strdup(result.c_str());
  } catch (const Ort::Exception &e) {
    ERROR_LOG("ORT Exception: " << e.what());
    if (error_out) {
      *error_out = strdup(e.what());
    }
    return nullptr;
  } catch (const std::exception &e) {
    ERROR_LOG("Standard Exception: " << e.what());
    if (error_out) {
      *error_out = strdup(e.what());
    }
    return nullptr;
  } catch (...) {
    ERROR_LOG("Unknown exception in ort_create_tensor");
    if (error_out) {
      *error_out = strdup("Unknown exception in ort_create_tensor");
    }
    return nullptr;
  }
}

// Convert an OrtValue to a different data type
extern "C" char *ort_convert_tensor(const char *value_id, const char *target_type, char **error_out) {

  try {
    auto it = g_ort_values.find(value_id);
    if (it == g_ort_values.end()) {
      if (error_out) {
        std::string error = "OrtValue with ID " + std::string(value_id) + " not found";
        *error_out = strdup(error.c_str());
      }
      return nullptr;
    }

    Ort::Value *tensor = it->second.get();

    // Get tensor info
    Ort::TypeInfo type_info = tensor->GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType current_type = tensor_info.GetElementType();
    std::vector<int64_t> shape = tensor_info.GetShape();

    // Determine target element type
    ONNXTensorElementDataType target_element_type;
    if (strcmp(target_type, "float32") == 0) {
      target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (strcmp(target_type, "float16") == 0) {
      target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    } else if (strcmp(target_type, "int32") == 0) {
      target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    } else if (strcmp(target_type, "int64") == 0) {
      target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } else if (strcmp(target_type, "uint8") == 0) {
      target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    } else if (strcmp(target_type, "bool") == 0) {
      target_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    } else {
      if (error_out) {
        std::string error = "Unsupported target data type: " + std::string(target_type);
        *error_out = strdup(error.c_str());
      }
      return nullptr;
    }

    // If already the target type, just return the current tensor
    if (current_type == target_element_type) {
      // Create result JSON
      std::string result = "{\"valueId\":\"" + std::string(value_id) + "\",\"dataType\":\"";
      result += target_type;
      result += "\",\"shape\":[";

      for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0)
          result += ",";
        result += std::to_string(shape[i]);
      }

      result += "],\"device\":\"cpu\"}";

      return strdup(result.c_str());
    }

    // Calculate total number of elements
    size_t total_elements = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      total_elements *= shape[i];
    }

    // Create memory info
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Create a new tensor with the converted data
    std::unique_ptr<Ort::Value> new_tensor = std::make_unique<Ort::Value>(nullptr);

    // Perform conversion based on source and target types
    if (current_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        target_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      // Float32 to Float16 conversion
      float *src_data = tensor->GetTensorMutableData<float>();

      // Allocate memory for the float16 data
      std::vector<uint16_t> float16_data(total_elements);

      // Convert each float32 to float16
      for (size_t i = 0; i < total_elements; i++) {
        float16_data[i] = Float16Utils::floatToFloat16(src_data[i]);
      }

      // Create new tensor with float16 data
      *new_tensor = Ort::Value::CreateTensor(memory_info, float16_data.data(), float16_data.size() * sizeof(uint16_t),
                                             shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    } else if (current_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
               target_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      // Float16 to Float32 conversion
      uint16_t *src_data = tensor->GetTensorMutableData<uint16_t>();

      // Allocate memory for the float32 data
      std::vector<float> float32_data(total_elements);

      // Convert each float16 to float32
      for (size_t i = 0; i < total_elements; i++) {
        float32_data[i] = Float16Utils::float16ToFloat(src_data[i]);
      }

      // Create new tensor with float32 data
      *new_tensor = Ort::Value::CreateTensor(memory_info, float32_data.data(), float32_data.size() * sizeof(float),
                                             shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    } else if (current_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
               target_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      // Float32 to Int32 conversion
      float *src_data = tensor->GetTensorMutableData<float>();

      // Allocate memory for the int32 data
      std::vector<int32_t> int32_data(total_elements);

      // Convert each float32 to int32
      for (size_t i = 0; i < total_elements; i++) {
        int32_data[i] = static_cast<int32_t>(src_data[i]);
      }

      // Create new tensor with int32 data
      *new_tensor = Ort::Value::CreateTensor(memory_info, int32_data.data(), int32_data.size() * sizeof(int32_t),
                                             shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    } else if (current_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
               target_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      // Int32 to Float32 conversion
      int32_t *src_data = tensor->GetTensorMutableData<int32_t>();

      // Allocate memory for the float32 data
      std::vector<float> float32_data(total_elements);

      // Convert each int32 to float32
      for (size_t i = 0; i < total_elements; i++) {
        float32_data[i] = static_cast<float>(src_data[i]);
      }

      // Create new tensor with float32 data
      *new_tensor = Ort::Value::CreateTensor(memory_info, float32_data.data(), float32_data.size() * sizeof(float),
                                             shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    } else if (current_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
               target_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      // Int32 to Int64 conversion
      int32_t *src_data = tensor->GetTensorMutableData<int32_t>();

      // Allocate memory for the int64 data
      std::vector<int64_t> int64_data(total_elements);

      // Convert each int32 to int64
      for (size_t i = 0; i < total_elements; i++) {
        int64_data[i] = static_cast<int64_t>(src_data[i]);
      }

      // Create new tensor with int64 data
      *new_tensor = Ort::Value::CreateTensor(memory_info, int64_data.data(), int64_data.size() * sizeof(int64_t),
                                             shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    } else if (current_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
               target_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      // Int64 to Int32 conversion
      int64_t *src_data = tensor->GetTensorMutableData<int64_t>();

      // Allocate memory for the int32 data
      std::vector<int32_t> int32_data(total_elements);

      // Convert each int64 to int32
      for (size_t i = 0; i < total_elements; i++) {
        int32_data[i] = static_cast<int32_t>(src_data[i]);
      }

      // Create new tensor with int32 data
      *new_tensor = Ort::Value::CreateTensor(memory_info, int32_data.data(), int32_data.size() * sizeof(int32_t),
                                             shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    } else {
      // Unsupported conversion
      if (error_out) {
        std::string error = "Unsupported conversion from ";
        switch (current_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          error += "float32";
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
          error += "float16";
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
          error += "int32";
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
          error += "int64";
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
          error += "uint8";
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
          error += "bool";
          break;
        default:
          error += "unknown";
          break;
        }
        error += " to " + std::string(target_type);
        *error_out = strdup(error.c_str());
      }
      return nullptr;
    }

    // Generate a new UUID for the new tensor
    std::string new_id = generate_ort_value_uuid();

    // Store the new tensor
    g_ort_values[new_id] = std::move(new_tensor);

    // Create result JSON
    std::string result = "{\"valueId\":\"" + new_id + "\",\"dataType\":\"";
    result += target_type;
    result += "\",\"shape\":[";

    for (size_t i = 0; i < shape.size(); i++) {
      if (i > 0)
        result += ",";
      result += std::to_string(shape[i]);
    }

    result += "],\"device\":\"cpu\"}";

    return strdup(result.c_str());
  } catch (const Ort::Exception &e) {
    if (error_out) {
      *error_out = strdup(e.what());
    }
    return nullptr;
  } catch (const std::exception &e) {
    if (error_out) {
      *error_out = strdup(e.what());
    }
    return nullptr;
  }
}

// Move an OrtValue to a different device
extern "C" char *ort_move_tensor_to_device(const char *value_id, const char *target_device, char **error_out) {
  try {
    auto it = g_ort_values.find(value_id);
    if (it == g_ort_values.end()) {
      if (error_out) {
        std::string error = "OrtValue with ID " + std::string(value_id) + " not found";
        *error_out = strdup(error.c_str());
      }
      return nullptr;
    }

    Ort::Value *tensor = it->second.get();

    // Get tensor info
    Ort::TypeInfo type_info = tensor->GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType element_type = tensor_info.GetElementType();
    std::vector<int64_t> shape = tensor_info.GetShape();

    // Currently we only support CPU, so just return the same tensor
    // In a full implementation, you would handle device transfer here
    if (strcmp(target_device, "cpu") != 0) {
      if (error_out) {
        std::string error = "Only CPU device is supported in this implementation";
        *error_out = strdup(error.c_str());
      }
      return nullptr;
    }

    // Get type string for response
    std::string type_str;
    switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      type_str = "float32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      type_str = "float16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      type_str = "int32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      type_str = "int64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      type_str = "uint8";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      type_str = "bool";
      break;
    default:
      type_str = "unknown";
      break;
    }

    // Create result JSON
    std::string result = "{\"valueId\":\"" + std::string(value_id) + "\",\"dataType\":\"";
    result += type_str;
    result += "\",\"shape\":[";

    for (size_t i = 0; i < shape.size(); i++) {
      if (i > 0)
        result += ",";
      result += std::to_string(shape[i]);
    }

    result += "],\"device\":\"cpu\"}";

    return strdup(result.c_str());
  } catch (const Ort::Exception &e) {
    if (error_out) {
      *error_out = strdup(e.what());
    }
    return nullptr;
  } catch (const std::exception &e) {
    if (error_out) {
      *error_out = strdup(e.what());
    }
    return nullptr;
  }
}

// Get data from an OrtValue
extern "C" char *ort_get_tensor_data(const char *value_id, const char *data_type, char **error_out) {
  try {
    auto it = g_ort_values.find(value_id);
    if (it == g_ort_values.end()) {
      if (error_out) {
        std::string error = "OrtValue with ID " + std::string(value_id) + " not found";
        *error_out = strdup(error.c_str());
      }
      return nullptr;
    }

    Ort::Value *tensor = it->second.get();

    // Get tensor info
    Ort::TypeInfo type_info = tensor->GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType element_type = tensor_info.GetElementType();
    std::vector<int64_t> shape = tensor_info.GetShape();

    // Calculate total number of elements
    size_t total_elements = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      total_elements *= shape[i];
    }

    // Create result JSON manually
    std::stringstream result;
    result << "{\"shape\":[";

    for (size_t i = 0; i < shape.size(); i++) {
      if (i > 0)
        result << ",";
      result << shape[i];
    }

    result << "],\"data\":[";

    // Extract data according to requested type
    try {
      if (strcmp(data_type, "float32") == 0) {
        // Handle float32 extraction/conversion
        if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          // Direct extraction if already float32
          float *tensor_data = tensor->GetTensorMutableData<float>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << tensor_data[i];
          }
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
          // Convert from float16
          uint16_t *tensor_data = tensor->GetTensorMutableData<uint16_t>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << Float16Utils::float16ToFloat(tensor_data[i]);
          }
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          // Convert from int32
          int32_t *tensor_data = tensor->GetTensorMutableData<int32_t>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << static_cast<float>(tensor_data[i]);
          }
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          // Convert from int64
          int64_t *tensor_data = tensor->GetTensorMutableData<int64_t>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << static_cast<float>(tensor_data[i]);
          }
        } else {
          throw std::runtime_error("Unsupported conversion to float32");
        }
      } else if (strcmp(data_type, "float16") == 0) {
        // Handle float16 extraction/conversion
        if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
          // Direct extraction if already float16
          uint16_t *tensor_data = tensor->GetTensorMutableData<uint16_t>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << static_cast<int>(tensor_data[i]);
          }
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          // Convert from float32
          float *tensor_data = tensor->GetTensorMutableData<float>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << static_cast<int>(Float16Utils::floatToFloat16(tensor_data[i]));
          }
        } else {
          throw std::runtime_error("Unsupported conversion to float16");
        }
      } else if (strcmp(data_type, "int32") == 0) {
        // Handle int32 extraction/conversion
        if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          // Direct extraction if already int32
          int32_t *tensor_data = tensor->GetTensorMutableData<int32_t>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << tensor_data[i];
          }
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          // Convert from float32
          float *tensor_data = tensor->GetTensorMutableData<float>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << static_cast<int32_t>(tensor_data[i]);
          }
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          // Convert from int64
          int64_t *tensor_data = tensor->GetTensorMutableData<int64_t>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << static_cast<int32_t>(tensor_data[i]);
          }
        } else {
          throw std::runtime_error("Unsupported conversion to int32");
        }
      } else if (strcmp(data_type, "int64") == 0) {
        // Handle int64 extraction/conversion
        if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          // Direct extraction if already int64
          int64_t *tensor_data = tensor->GetTensorMutableData<int64_t>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << tensor_data[i];
          }
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          // Convert from float32
          float *tensor_data = tensor->GetTensorMutableData<float>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << static_cast<int64_t>(tensor_data[i]);
          }
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          // Convert from int32
          int32_t *tensor_data = tensor->GetTensorMutableData<int32_t>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << static_cast<int64_t>(tensor_data[i]);
          }
        } else {
          throw std::runtime_error("Unsupported conversion to int64");
        }
      } else if (strcmp(data_type, "uint8") == 0) {
        // Handle uint8 extraction/conversion
        if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
          // Direct extraction if already uint8
          uint8_t *tensor_data = tensor->GetTensorMutableData<uint8_t>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << static_cast<int>(tensor_data[i]);
          }
        } else {
          throw std::runtime_error("Unsupported conversion to uint8");
        }
      } else if (strcmp(data_type, "bool") == 0) {
        // Handle bool extraction/conversion
        if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
          // Direct extraction if already bool
          bool *tensor_data = tensor->GetTensorMutableData<bool>();
          for (size_t i = 0; i < total_elements; i++) {
            if (i > 0)
              result << ",";
            result << (tensor_data[i] ? "true" : "false");
          }
        } else {
          throw std::runtime_error("Unsupported conversion to bool");
        }
      } else {
        throw std::runtime_error(std::string("Unsupported data type: ") + data_type);
      }
    } catch (const std::exception &e) {
      if (error_out) {
        std::string error = std::string("Data extraction error: ") + e.what();
        *error_out = strdup(error.c_str());
      }
      return nullptr;
    }

    result << "]}";
    return strdup(result.str().c_str());
  } catch (const Ort::Exception &e) {
    if (error_out) {
      *error_out = strdup(e.what());
    }
    return nullptr;
  } catch (const std::exception &e) {
    if (error_out) {
      *error_out = strdup(e.what());
    }
    return nullptr;
  }
}

// Release an OrtValue
extern "C" bool ort_release_tensor(const char *value_id, char **error_out) {
  try {
    auto it = g_ort_values.find(value_id);
    if (it == g_ort_values.end()) {
      if (error_out) {
        std::string error = "OrtValue with ID " + std::string(value_id) + " not found";
        *error_out = strdup(error.c_str());
      }
      return false;
    }

    g_ort_values.erase(it);
    return true;
  } catch (const std::exception &e) {
    if (error_out) {
      *error_out = strdup(e.what());
    }
    return false;
  }
}

// Helper function to generate a UUID
// This is a simple implementation, you may want to use a proper UUID library
std::string generate_ort_value_uuid() {
  static int counter = 0;
  return "tensor_" + std::to_string(counter++);
}