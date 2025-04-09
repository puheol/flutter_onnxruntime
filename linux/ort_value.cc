#include "ort_value.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Maintain global mappings for OrtValue objects
std::unordered_map<std::string, std::unique_ptr<Ort::Value>> g_ort_values;

// Create a new OrtValue from data
extern "C" char *ort_create_tensor(const char *source_type, const void *data, const int64_t *shape, int shape_len,
                                   const char *target_type, const char *device, char **error_out) {

  try {
    // Create ORT environment and allocator if not initialized
    static Ort::Env *env = nullptr;
    static Ort::AllocatorWithDefaultOptions allocator;
    if (env == nullptr) {
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

    // Create memory info for the target device
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Create tensor with data
    size_t data_size = total_elements * element_size;

    // Choose appropriate conversion based on target type
    ONNXTensorElementDataType target_element_type = element_type;
    if (target_type && *target_type) {
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

    // Create the tensor
    std::unique_ptr<Ort::Value> tensor = std::make_unique<Ort::Value>();

    // Copy and potentially convert the data
    void *new_data = malloc(data_size);
    if (!new_data) {
      if (error_out) {
        *error_out = strdup("Failed to allocate memory for tensor data");
      }
      return nullptr;
    }

    // First copy the data as-is
    memcpy(new_data, data, data_size);

    // Create the tensor with the copied data
    Ort::Value ort_tensor =
        Ort::Value::CreateTensor(memory_info, new_data, data_size, shape_vec.data(), shape_vec.size(), element_type);

    // If target type differs from source type, perform conversion
    if (target_element_type != element_type) {
      // In a complete implementation, you would convert between types here
      // This would use ONNX Runtime APIs for type conversion
      // For now, we'll just log the conversion request
      std::cout << "Requested conversion from " << source_type << " to " << target_type
                << " (not implemented in this example)" << std::endl;
    }

    // Store the OrtValue
    std::string id = generate_ort_value_uuid();
    g_ort_values[id] = std::move(tensor);

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

    // In a real implementation, you would convert between types here
    // This would use ONNX Runtime APIs for type conversion
    // For now, we'll just create a new tensor ID for the "converted" tensor

    // Generate a new UUID for the "converted" tensor
    std::string new_id = generate_ort_value_uuid();

    // For demo purposes, we're not actually converting, just making a new reference
    g_ort_values[new_id] = std::move(it->second);

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

    // Currently, we only support CPU
    if (strcmp(target_device, "cpu") != 0) {
      if (error_out) {
        std::string error = "Only CPU device is supported in this implementation";
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

    // Get data type string
    std::string data_type;
    switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      data_type = "float32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      data_type = "float16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      data_type = "int32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      data_type = "int64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      data_type = "uint8";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      data_type = "bool";
      break;
    default:
      data_type = "unknown";
      break;
    }

    // Create result JSON
    std::string result = "{\"valueId\":\"" + std::string(value_id) + "\",\"dataType\":\"";
    result += data_type;
    result += "\",\"shape\":[";

    for (size_t i = 0; i < shape.size(); i++) {
      if (i > 0)
        result += ",";
      result += std::to_string(shape[i]);
    }

    result += "],\"device\":\"";
    result += target_device;
    result += "\"}";

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

    // Get data based on requested type
    std::string result = "{\"data\":[";

    if (strcmp(data_type, "float32") == 0) {
      if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        // Direct access
        float *data = tensor->GetTensorMutableData<float>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(data[i]);
        }
      } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        // Convert int32 to float32
        int32_t *data = tensor->GetTensorMutableData<int32_t>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(static_cast<float>(data[i]));
        }
      } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        // Convert int64 to float32
        int64_t *data = tensor->GetTensorMutableData<int64_t>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(static_cast<float>(data[i]));
        }
      } else {
        if (error_out) {
          *error_out = strdup("Cannot convert to float32");
        }
        return nullptr;
      }
    } else if (strcmp(data_type, "int32") == 0) {
      if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        // Direct access
        int32_t *data = tensor->GetTensorMutableData<int32_t>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(data[i]);
        }
      } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        // Convert float to int32
        float *data = tensor->GetTensorMutableData<float>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(static_cast<int32_t>(data[i]));
        }
      } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        // Convert int64 to int32
        int64_t *data = tensor->GetTensorMutableData<int64_t>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(static_cast<int32_t>(data[i]));
        }
      } else {
        if (error_out) {
          *error_out = strdup("Cannot convert to int32");
        }
        return nullptr;
      }
    } else if (strcmp(data_type, "int64") == 0) {
      if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        // Direct access
        int64_t *data = tensor->GetTensorMutableData<int64_t>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(data[i]);
        }
      } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        // Convert float to int64
        float *data = tensor->GetTensorMutableData<float>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(static_cast<int64_t>(data[i]));
        }
      } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        // Convert int32 to int64
        int32_t *data = tensor->GetTensorMutableData<int32_t>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(static_cast<int64_t>(data[i]));
        }
      } else {
        if (error_out) {
          *error_out = strdup("Cannot convert to int64");
        }
        return nullptr;
      }
    } else if (strcmp(data_type, "uint8") == 0) {
      if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        // Direct access
        uint8_t *data = tensor->GetTensorMutableData<uint8_t>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += std::to_string(data[i]);
        }
      } else {
        if (error_out) {
          *error_out = strdup("Cannot convert to uint8");
        }
        return nullptr;
      }
    } else if (strcmp(data_type, "bool") == 0) {
      if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        // Direct access
        bool *data = tensor->GetTensorMutableData<bool>();
        for (size_t i = 0; i < total_elements; i++) {
          if (i > 0)
            result += ",";
          result += data[i] ? "true" : "false";
        }
      } else {
        if (error_out) {
          *error_out = strdup("Cannot convert to bool");
        }
        return nullptr;
      }
    } else {
      if (error_out) {
        std::string error = "Unsupported data type: " + std::string(data_type);
        *error_out = strdup(error.c_str());
      }
      return nullptr;
    }

    result += "],\"shape\":[";

    for (size_t i = 0; i < shape.size(); i++) {
      if (i > 0)
        result += ",";
      result += std::to_string(shape[i]);
    }

    result += "]}";

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