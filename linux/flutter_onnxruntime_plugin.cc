#include "include/flutter_onnxruntime/flutter_onnxruntime_plugin.h"

#include <flutter_linux/flutter_linux.h>
#include <gtk/gtk.h>
#include <sys/utsname.h>

#include <cstdlib> // For strdup
#include <cstring>
#include <iomanip>  // Added for std::ostringstream and std::setprecision
#include <iostream> // Added for debug logging
#include <map>
#include <memory>
#include <string>
#include <vector>

// Include ONNX Runtime headers
#include "ort_value.h"
#include <onnxruntime_cxx_api.h>

#include "flutter_onnxruntime_plugin_private.h"

// Structure to hold session information
struct SessionInfo {
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

// Map to store active sessions
static std::map<std::string, SessionInfo> g_sessions;
static Ort::Env g_ort_env{ORT_LOGGING_LEVEL_VERBOSE, "flutter_onnxruntime"};

// For debugging
#define DEBUG_LOG(msg) std::cout << "[DEBUG] " << msg << std::endl

#define FLUTTER_ONNXRUNTIME_PLUGIN(obj)                                                                                \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), flutter_onnxruntime_plugin_get_type(), FlutterOnnxruntimePlugin))

struct _FlutterOnnxruntimePlugin {
  GObject parent_instance;
};

G_DEFINE_TYPE(FlutterOnnxruntimePlugin, flutter_onnxruntime_plugin, g_object_get_type())

// Helper function to generate a UUID
std::string generate_session_uuid() {
  static int counter = 0;
  return "session_" + std::to_string(time(nullptr)) + "_" + std::to_string(counter++);
}

// Helper function to convert FlValue to std::vector
template <typename T> std::vector<T> fl_value_to_vector(FlValue *value) {
  std::vector<T> result;
  if (fl_value_get_type(value) == FL_VALUE_TYPE_LIST) {
    size_t length = fl_value_get_length(value);
    result.reserve(length); // Pre-allocate memory for better performance
    for (size_t i = 0; i < length; i++) {
      FlValue *item = fl_value_get_list_value(value, i); // This function does not transfer ownership

      if (std::is_same<T, int64_t>::value) {
        result.push_back(static_cast<T>(fl_value_get_int(item)));
      } else if (std::is_same<T, float>::value) {
        // Check if the item is a float or int
        FlValueType item_type = fl_value_get_type(item);
        if (item_type == FL_VALUE_TYPE_FLOAT) {
          float value = fl_value_get_float(item);
          result.push_back(static_cast<T>(value));
        } else if (item_type == FL_VALUE_TYPE_INT) {
          // If it's an integer, convert to float
          int64_t int_value = fl_value_get_int(item);
          float float_value = static_cast<float>(int_value);
          result.push_back(static_cast<T>(float_value));
        } else {
          // Default to 0 for unsupported types
          result.push_back(static_cast<T>(0.0f));
        }
      }
      // Note: No need to unref item as fl_value_get_list_value doesn't transfer
      // ownership
    }
  }
  return result;
}

// Helper function to convert FlValue map to std::map
std::map<std::string, FlValue *> fl_value_to_map(FlValue *value) {
  std::map<std::string, FlValue *> result;
  if (fl_value_get_type(value) == FL_VALUE_TYPE_MAP) {
    size_t length = fl_value_get_length(value);
    for (size_t i = 0; i < length; i++) {
      // Get key and value at index i - these functions don't transfer ownership
      FlValue *key = fl_value_get_map_key(value, i);
      if (fl_value_get_type(key) == FL_VALUE_TYPE_STRING) {
        std::string key_str = fl_value_get_string(key);
        FlValue *val = fl_value_get_map_value(value, i);
        // Store the pointer without taking ownership
        result[key_str] = val;
      }
      // No need to unref key or val as the get functions don't transfer
      // ownership
    }
  }
  return result;
}

// Helper function to convert std::vector to FlValue list
template <typename T> FlValue *vector_to_fl_value(const std::vector<T> &vec) {
  // This is a generic template - should be specialized for each type
  return fl_value_new_list();
}

// Template specialization for strings
template <> FlValue *vector_to_fl_value<std::string>(const std::vector<std::string> &vec) {
  FlValue *list = fl_value_new_list();
  for (const auto &item : vec) {
    fl_value_append_take(list, fl_value_new_string(item.c_str()));
  }
  return list;
}

// Template specialization for int64_t
template <> FlValue *vector_to_fl_value<int64_t>(const std::vector<int64_t> &vec) {
  FlValue *list = fl_value_new_list();
  for (const auto &item : vec) {
    fl_value_append_take(list, fl_value_new_int(item));
  }
  return list;
}

// Template specialization for float
template <> FlValue *vector_to_fl_value<float>(const std::vector<float> &vec) {
  FlValue *list = fl_value_new_list();
  for (const auto &item : vec) {
    fl_value_append_take(list, fl_value_new_float(item));
  }
  return list;
}

// Template specialization for int
template <> FlValue *vector_to_fl_value<int>(const std::vector<int> &vec) {
  FlValue *list = fl_value_new_list();
  for (const auto &item : vec) {
    fl_value_append_take(list, fl_value_new_int(item));
  }
  return list;
}

// Called when a method call is received from Flutter.
static void flutter_onnxruntime_plugin_handle_method_call(FlutterOnnxruntimePlugin *self, FlMethodCall *method_call) {
  g_autoptr(FlMethodResponse) response = nullptr;

  const gchar *method = fl_method_call_get_name(method_call);
  g_autoptr(FlValue) args = fl_method_call_get_args(method_call);

  if (strcmp(method, "getPlatformVersion") == 0) {
    response = get_platform_version();
  } else if (strcmp(method, "createSession") == 0) {
    // Extract arguments
    FlValue *model_path_value = fl_value_lookup_string(args, "modelPath");
    FlValue *session_options_value = fl_value_lookup_string(args, "sessionOptions");

    if (model_path_value == nullptr || fl_value_get_type(model_path_value) != FL_VALUE_TYPE_STRING) {
      response =
          FL_METHOD_RESPONSE(fl_method_error_response_new("NULL_MODEL_PATH", "Model path cannot be null", nullptr));
    } else {
      const char *model_path = fl_value_get_string(model_path_value);

      try {
        // Create session options
        Ort::SessionOptions session_options;

        // Configure session options if provided
        if (session_options_value != nullptr && fl_value_get_type(session_options_value) == FL_VALUE_TYPE_MAP) {
          auto options_map = fl_value_to_map(session_options_value);

          // Set threading options
          auto intra_threads_val = options_map.find("intraOpNumThreads");
          if (intra_threads_val != options_map.end() &&
              fl_value_get_type(intra_threads_val->second) == FL_VALUE_TYPE_INT) {
            session_options.SetIntraOpNumThreads(fl_value_get_int(intra_threads_val->second));
          }

          auto inter_threads_val = options_map.find("interOpNumThreads");
          if (inter_threads_val != options_map.end() &&
              fl_value_get_type(inter_threads_val->second) == FL_VALUE_TYPE_INT) {
            session_options.SetInterOpNumThreads(fl_value_get_int(inter_threads_val->second));
          }
        }

        // Create session
        std::unique_ptr<Ort::Session> ort_session =
            std::make_unique<Ort::Session>(g_ort_env, model_path, session_options);

        // Generate a session ID
        std::string session_id = generate_session_uuid();

        // Get input names
        std::vector<std::string> input_names;
        size_t num_inputs = ort_session->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
          Ort::AllocatorWithDefaultOptions allocator;
          // Get the name using the API that returns an allocated string
          auto input_name = ort_session->GetInputNameAllocated(i, allocator);
          input_names.push_back(std::string(input_name.get()));
        }

        // Get output names
        std::vector<std::string> output_names;
        size_t num_outputs = ort_session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
          Ort::AllocatorWithDefaultOptions allocator;
          // Get output name using the API that returns an allocated string
          auto output_name = ort_session->GetOutputNameAllocated(i, allocator);
          output_names.push_back(std::string(output_name.get()));
        }

        // Store session info
        SessionInfo session_info;
        session_info.session = std::move(ort_session);
        session_info.input_names = input_names;
        session_info.output_names = output_names;

        g_sessions[session_id] = std::move(session_info);

        // Create response manually (removed g_autoptr)
        FlValue *result = fl_value_new_map(); // result ref count = 1
        fl_value_set_string_take(result, "sessionId", fl_value_new_string(session_id.c_str()));

        // Create values and add them to the map
        FlValue *input_names_value = vector_to_fl_value<std::string>(input_names);
        fl_value_set_string_take(result, "inputNames", input_names_value);

        FlValue *output_names_value = vector_to_fl_value<std::string>(output_names);
        fl_value_set_string_take(result, "outputNames", output_names_value);

        response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
      } catch (const Ort::Exception &e) {
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("ORT_ERROR", e.what(), nullptr));
      } catch (const std::exception &e) {
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("GENERIC_ERROR", e.what(), nullptr));
      }
    }
  } else if (strcmp(method, "runInference") == 0) {
    DEBUG_LOG("Running inference");
    // Extract arguments
    FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");
    FlValue *inputs_value = fl_value_lookup_string(args, "inputs");

    if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
      response =
          FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session ID cannot be null", nullptr));
    } else if (inputs_value == nullptr || fl_value_get_type(inputs_value) != FL_VALUE_TYPE_MAP) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("NULL_INPUTS", "Inputs cannot be null", nullptr));
    } else {
      const char *session_id = fl_value_get_string(session_id_value);

      // Check if session exists
      auto session_it = g_sessions.find(session_id);
      if (session_it == g_sessions.end()) {
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      } else {
        try {
          // Process inference
          // This is a simplified implementation - for a complete solution,
          // you would need to handle all tensor types and shapes

          // Prepare input tensors
          auto &session_info = session_it->second;
          auto inputs_map = fl_value_to_map(inputs_value);

          std::vector<Ort::Value> ort_inputs;
          std::vector<const char *> input_names_cstr;
          // store input buffers unless the values will be cleaned after the
          // current iteration
          std::vector<std::vector<float>> input_buffers;

          // For simplicity, we'll just handle float inputs in this example
          for (const auto &name : session_info.input_names) {
            auto it = inputs_map.find(name);

            if (it != inputs_map.end()) {
              // Check if this is an OrtValue reference
              if (fl_value_get_type(it->second) == FL_VALUE_TYPE_MAP) {
                // Check for valueId in the map
                FlValue *value_id_val = fl_value_lookup_string(it->second, "valueId");
                if (value_id_val != nullptr && fl_value_get_type(value_id_val) == FL_VALUE_TYPE_STRING) {
                  const char *value_id = fl_value_get_string(value_id_val);
                  // Look up the OrtValue in the global map
                  auto ort_value_it = g_ort_values.find(value_id);
                  if (ort_value_it != g_ort_values.end()) {
                    // Create a completely new tensor
                    // This is a simplified implementation

                    // First check if the tensor exists
                    if (ort_value_it->second && ort_value_it->second->IsTensor()) {
                      Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                                               OrtMemType::OrtMemTypeDefault);

                      // We'll use a dummy tensor for now
                      // In a real implementation, you'd extract the data and shape from the original
                      std::vector<float> dummy_data = {1.0f};
                      std::vector<int64_t> dummy_shape = {1};

                      Ort::Value tensor = Ort::Value::CreateTensor<float>(
                          memory_info, dummy_data.data(), dummy_data.size(), dummy_shape.data(), dummy_shape.size());

                      ort_inputs.push_back(std::move(tensor));
                      input_names_cstr.push_back(name.c_str());
                      continue;
                    } else {
                      // Handle invalid tensor
                      response = FL_METHOD_RESPONSE(fl_method_error_response_new(
                          "INVALID_TENSOR", "The referenced tensor is not valid", nullptr));
                      return;
                    }
                  } else {
                    // OrtValue not found
                    response = FL_METHOD_RESPONSE(fl_method_error_response_new(
                        "INVALID_ORT_VALUE", ("OrtValue with ID " + std::string(value_id) + " not found").c_str(),
                        nullptr));
                    return;
                  }
                }
              }

              // Handle regular list input if not an OrtValue
              if (fl_value_get_type(it->second) == FL_VALUE_TYPE_LIST) {
                // Get shape if provided
                std::vector<int64_t> shape;
                auto shape_it = inputs_map.find(name + "_shape");
                if (shape_it != inputs_map.end() && fl_value_get_type(shape_it->second) == FL_VALUE_TYPE_LIST) {
                  shape = fl_value_to_vector<int64_t>(shape_it->second);
                } else {
                  // Default shape: just a single dimension with the list length
                  shape.push_back(fl_value_get_length(it->second));
                }

                // Convert to float vector and persist it in the container
                std::vector<float> data = fl_value_to_vector<float>(it->second);
                input_buffers.push_back(std::move(data));
                auto &buffer = input_buffers.back();

                // Create tensor
                Ort::MemoryInfo memory_info =
                    Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

                Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, buffer.data(), buffer.size(),
                                                                    shape.data(), shape.size());

                ort_inputs.push_back(std::move(tensor));
                input_names_cstr.push_back(name.c_str());
              }
            }
          }

          // Prepare output names
          std::vector<const char *> output_names_cstr;
          for (const auto &name : session_info.output_names) {
            output_names_cstr.push_back(name.c_str());
          }

          // Run inference
          auto output_tensors =
              session_info.session->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), ort_inputs.data(),
                                        ort_inputs.size(), output_names_cstr.data(), output_names_cstr.size());

          // Create the outputs map manually (removed g_autoptr)
          FlValue *result_map = fl_value_new_map();  // ref count = 1
          FlValue *outputs_map = fl_value_new_map(); // ref count = 1

          // Process outputs safely
          for (size_t i = 0; i < output_tensors.size(); i++) {
            const auto &output_name = session_info.output_names[i];
            auto &tensor = output_tensors[i];

            // Get tensor info
            Ort::TypeInfo type_info = tensor.GetTypeInfo();
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType tensor_type = tensor_info.GetElementType();
            auto shape = tensor_info.GetShape();

            FlValue *shape_list = vector_to_fl_value<int64_t>(shape); // shape_list ref=1
            fl_value_set_string(outputs_map, (output_name + "_shape").c_str(), shape_list);
            fl_value_unref(shape_list); // Decrement ref count after adding to map

            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
              // Calculate flat size
              size_t flat_size = 1;
              for (auto dim : shape) {
                flat_size *= (dim > 0) ? dim : 1; // Handle dynamic dimensions
              }

              // Get float data directly from the output tensor in the vector
              const float *data = tensor.GetTensorData<float>();
              if (data != nullptr && tensor.IsTensor()) {
                // Copy data into a vector
                std::vector<float> data_vec(data, data + flat_size);

                FlValue *data_list = vector_to_fl_value<float>(data_vec); // data_list ref=1
                fl_value_set_string(outputs_map, output_name.c_str(), data_list);
                fl_value_unref(data_list); // Decrement ref count after adding to map
              }
            } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
              // Calculate flat size
              size_t flat_size = 1;
              for (auto dim : shape) {
                flat_size *= (dim > 0) ? dim : 1; // Handle dynamic dimensions
              }

              // Get int data directly
              const int32_t *data = tensor.GetTensorData<int32_t>();
              if (data != nullptr) {
                std::vector<int> data_vec(data, data + flat_size);

                FlValue *data_list = vector_to_fl_value<int>(data_vec); // data_list ref=1
                fl_value_set_string(outputs_map, output_name.c_str(), data_list);
                fl_value_unref(data_list); // Decrement ref count after adding to map
              }
            }
            // Add more data types as needed
          }

          // Set the outputs map in the result
          fl_value_set_string(result_map, "outputs", outputs_map);

          // Return the result
          response = FL_METHOD_RESPONSE(fl_method_success_response_new(result_map));
        } catch (const Ort::Exception &e) {
          response = FL_METHOD_RESPONSE(fl_method_error_response_new("ORT_ERROR", e.what(), nullptr));
        } catch (const std::exception &e) {
          response = FL_METHOD_RESPONSE(fl_method_error_response_new("GENERIC_ERROR", e.what(), nullptr));
        }
      }
    }
  } else if (strcmp(method, "closeSession") == 0) {
    // Extract arguments
    FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

    if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
      response =
          FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session ID cannot be null", nullptr));
    } else {
      const char *session_id = fl_value_get_string(session_id_value);

      // Check if session exists
      auto session_it = g_sessions.find(session_id);
      if (session_it == g_sessions.end()) {
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      } else {
        try {
          // Remove session
          g_sessions.erase(session_it);

          response = FL_METHOD_RESPONSE(fl_method_success_response_new(nullptr));
        } catch (const std::exception &e) {
          response = FL_METHOD_RESPONSE(fl_method_error_response_new("GENERIC_ERROR", e.what(), nullptr));
        }
      }
    }
  } else if (strcmp(method, "getMetadata") == 0) {
    // Extract the session ID
    FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

    if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    const char *session_id = fl_value_get_string(session_id_value);

    // Check if the session exists
    if (g_sessions.find(session_id) == g_sessions.end()) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    try {
      // Get the session
      auto &session_info = g_sessions[session_id];
      auto &session = session_info.session;

      // Get metadata for the model
      Ort::ModelMetadata model_metadata = session->GetModelMetadata();
      Ort::AllocatorWithDefaultOptions allocator;

      // Extract metadata details
      auto producer_name = model_metadata.GetProducerNameAllocated(allocator);
      auto graph_name = model_metadata.GetGraphNameAllocated(allocator);
      auto domain = model_metadata.GetDomainAllocated(allocator);
      auto description = model_metadata.GetDescriptionAllocated(allocator);
      int64_t version = model_metadata.GetVersion();

      // Create empty custom metadata map - different ORT versions have different APIs
      FlValue *custom_metadata_map = fl_value_new_map();

      // Create response
      FlValue *result = fl_value_new_map();
      fl_value_set_string_take(result, "producerName", fl_value_new_string(producer_name.get()));
      fl_value_set_string_take(result, "graphName", fl_value_new_string(graph_name.get()));
      fl_value_set_string_take(result, "domain", fl_value_new_string(domain.get()));
      fl_value_set_string_take(result, "description", fl_value_new_string(description.get()));
      fl_value_set_string_take(result, "version", fl_value_new_int(version));
      fl_value_set_string_take(result, "customMetadataMap", custom_metadata_map);

      response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    } catch (const Ort::Exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("ORT_ERROR", e.what(), nullptr));
    } catch (const std::exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("GENERIC_ERROR", e.what(), nullptr));
    }
  } else if (strcmp(method, "getInputInfo") == 0) {
    // Extract the session ID
    FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

    if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    const char *session_id = fl_value_get_string(session_id_value);

    // Check if the session exists
    if (g_sessions.find(session_id) == g_sessions.end()) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    try {
      // Get the session
      auto &session_info = g_sessions[session_id];
      auto &session = session_info.session;

      // Create result list
      FlValue *node_info_list = fl_value_new_list();

      // Get all input info
      size_t num_inputs = session->GetInputCount();
      Ort::AllocatorWithDefaultOptions allocator;

      for (size_t i = 0; i < num_inputs; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        auto type_info = session->GetInputTypeInfo(i);

        FlValue *info_map = fl_value_new_map();
        fl_value_set_string_take(info_map, "name", fl_value_new_string(input_name.get()));

        // Check if it's a tensor type
        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
          auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
          auto shape = tensor_info.GetShape();

          // Convert shape to list
          FlValue *shape_list = fl_value_new_list();
          for (const auto &dim : shape) {
            fl_value_append_take(shape_list, fl_value_new_int(dim));
          }
          fl_value_set_string_take(info_map, "shape", shape_list);

          // Add type info
          ONNXTensorElementDataType element_type = tensor_info.GetElementType();
          const char *type_str;
          switch (element_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            type_str = "FLOAT";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            type_str = "UINT8";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            type_str = "INT8";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            type_str = "UINT16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            type_str = "INT16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            type_str = "INT32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            type_str = "INT64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            type_str = "STRING";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            type_str = "BOOL";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            type_str = "FLOAT16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            type_str = "DOUBLE";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            type_str = "UINT32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            type_str = "UINT64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            type_str = "COMPLEX64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            type_str = "COMPLEX128";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            type_str = "BFLOAT16";
            break;
          default:
            type_str = "UNKNOWN";
            break;
          }
          fl_value_set_string_take(info_map, "type", fl_value_new_string(type_str));
        } else {
          // For non-tensor types, provide an empty shape
          FlValue *empty_shape = fl_value_new_list();
          fl_value_set_string_take(info_map, "shape", empty_shape);
        }

        fl_value_append_take(node_info_list, info_map);
      }

      response = FL_METHOD_RESPONSE(fl_method_success_response_new(node_info_list));
    } catch (const Ort::Exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("ORT_ERROR", e.what(), nullptr));
    } catch (const std::exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("GENERIC_ERROR", e.what(), nullptr));
    }
  } else if (strcmp(method, "getOutputInfo") == 0) {
    // Extract the session ID
    FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

    if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    const char *session_id = fl_value_get_string(session_id_value);

    // Check if the session exists
    if (g_sessions.find(session_id) == g_sessions.end()) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    try {
      // Get the session
      auto &session_info = g_sessions[session_id];
      auto &session = session_info.session;

      // Create result list
      FlValue *node_info_list = fl_value_new_list();

      // Get all output info
      size_t num_outputs = session->GetOutputCount();
      Ort::AllocatorWithDefaultOptions allocator;

      for (size_t i = 0; i < num_outputs; i++) {
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        auto type_info = session->GetOutputTypeInfo(i);

        FlValue *info_map = fl_value_new_map();
        fl_value_set_string_take(info_map, "name", fl_value_new_string(output_name.get()));

        // Check if it's a tensor type
        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
          auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
          auto shape = tensor_info.GetShape();

          // Convert shape to list
          FlValue *shape_list = fl_value_new_list();
          for (const auto &dim : shape) {
            fl_value_append_take(shape_list, fl_value_new_int(dim));
          }
          fl_value_set_string_take(info_map, "shape", shape_list);

          // Add type info
          ONNXTensorElementDataType element_type = tensor_info.GetElementType();
          const char *type_str;
          switch (element_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            type_str = "FLOAT";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            type_str = "UINT8";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            type_str = "INT8";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            type_str = "UINT16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            type_str = "INT16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            type_str = "INT32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            type_str = "INT64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            type_str = "STRING";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            type_str = "BOOL";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            type_str = "FLOAT16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            type_str = "DOUBLE";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            type_str = "UINT32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            type_str = "UINT64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            type_str = "COMPLEX64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            type_str = "COMPLEX128";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            type_str = "BFLOAT16";
            break;
          default:
            type_str = "UNKNOWN";
            break;
          }
          fl_value_set_string_take(info_map, "type", fl_value_new_string(type_str));
        } else {
          // For non-tensor types, provide an empty shape
          FlValue *empty_shape = fl_value_new_list();
          fl_value_set_string_take(info_map, "shape", empty_shape);
        }

        fl_value_append_take(node_info_list, info_map);
      }

      response = FL_METHOD_RESPONSE(fl_method_success_response_new(node_info_list));
    } catch (const Ort::Exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("ORT_ERROR", e.what(), nullptr));
    } catch (const std::exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("GENERIC_ERROR", e.what(), nullptr));
    }
  } else {
    response = FL_METHOD_RESPONSE(fl_method_not_implemented_response_new());
  }

  fl_method_call_respond(method_call, response, nullptr);
}

FlMethodResponse *get_platform_version() {
  struct utsname uname_data = {};
  uname(&uname_data);
  g_autofree gchar *version = g_strdup_printf("Linux %s", uname_data.version);
  g_autoptr(FlValue) result = fl_value_new_string(version);
  return FL_METHOD_RESPONSE(fl_method_success_response_new(result));
}

static void flutter_onnxruntime_plugin_dispose(GObject *object) {
  G_OBJECT_CLASS(flutter_onnxruntime_plugin_parent_class)->dispose(object);
}

static void flutter_onnxruntime_plugin_class_init(FlutterOnnxruntimePluginClass *klass) {
  G_OBJECT_CLASS(klass)->dispose = flutter_onnxruntime_plugin_dispose;
}

static void flutter_onnxruntime_plugin_init(FlutterOnnxruntimePlugin *self) {}

static void method_call_cb(FlMethodChannel *channel, FlMethodCall *method_call, gpointer user_data) {
  FlutterOnnxruntimePlugin *plugin = FLUTTER_ONNXRUNTIME_PLUGIN(user_data);
  flutter_onnxruntime_plugin_handle_method_call(plugin, method_call);
}

void flutter_onnxruntime_plugin_register_with_registrar(FlPluginRegistrar *registrar) {
  FlutterOnnxruntimePlugin *plugin =
      FLUTTER_ONNXRUNTIME_PLUGIN(g_object_new(flutter_onnxruntime_plugin_get_type(), nullptr));

  g_autoptr(FlStandardMethodCodec) codec = fl_standard_method_codec_new();
  g_autoptr(FlMethodChannel) channel = fl_method_channel_new(fl_plugin_registrar_get_messenger(registrar),
                                                             "flutter_onnxruntime", FL_METHOD_CODEC(codec));
  fl_method_channel_set_method_call_handler(channel, method_call_cb, g_object_ref(plugin), g_object_unref);

  g_object_unref(plugin);
}
