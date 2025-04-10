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
#define ERROR_LOG(msg) std::cerr << "[ERROR] " << msg << std::endl
#define INFO_LOG(msg) std::cout << "[INFO] " << msg << std::endl

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

// Helper function to get readable string for FlValueType
std::string fl_value_type_to_string(FlValueType type) {
  switch (type) {
  case FL_VALUE_TYPE_NULL:
    return "FL_VALUE_TYPE_NULL";
  case FL_VALUE_TYPE_BOOL:
    return "FL_VALUE_TYPE_BOOL";
  case FL_VALUE_TYPE_INT:
    return "FL_VALUE_TYPE_INT";
  case FL_VALUE_TYPE_FLOAT:
    return "FL_VALUE_TYPE_FLOAT";
  case FL_VALUE_TYPE_STRING:
    return "FL_VALUE_TYPE_STRING";
  case FL_VALUE_TYPE_UINT8_LIST:
    return "FL_VALUE_TYPE_UINT8_LIST";
  case FL_VALUE_TYPE_INT32_LIST:
    return "FL_VALUE_TYPE_INT32_LIST";
  case FL_VALUE_TYPE_INT64_LIST:
    return "FL_VALUE_TYPE_INT64_LIST";
  case FL_VALUE_TYPE_FLOAT_LIST:
    return "FL_VALUE_TYPE_FLOAT_LIST";
  case FL_VALUE_TYPE_LIST:
    return "FL_VALUE_TYPE_LIST";
  case FL_VALUE_TYPE_MAP:
    return "FL_VALUE_TYPE_MAP";
  default:
    // Handle special types not in the enum
    int type_val = static_cast<int>(type);
    if (type_val == 11)
      return "FL_VALUE_TYPE_TYPED_DATA_FLOAT32";
    if (type_val == 12)
      return "FL_VALUE_TYPE_TYPED_DATA_INT32";
    if (type_val == 13)
      return "FL_VALUE_TYPE_TYPED_DATA_INT64";
    return "UNKNOWN_TYPE(" + std::to_string(type_val) + ")";
  }
}

// Helper function to convert FlValue to std::vector
template <typename T> std::vector<T> fl_value_to_vector(FlValue *value) {
  std::vector<T> result;
  FlValueType data_type = fl_value_get_type(value);
  int data_type_val = static_cast<int>(data_type);
  if (data_type_val == 9) { // FL_VALUE_TYPE_LIST
    size_t length = fl_value_get_length(value);
    result.reserve(length); // Pre-allocate memory for better performance
    for (size_t i = 0; i < length; i++) {
      FlValue *item = fl_value_get_list_value(value, i); // This function does not transfer ownership

      if (std::is_same<T, int64_t>::value) {
        result.push_back(static_cast<T>(fl_value_get_int(item)));
      } else if (std::is_same<T, int>::value) {
        // Handle conversion to int explicitly (for int32 data)
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

  DEBUG_LOG("Received method call: " << method);

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
      ERROR_LOG("Invalid session ID");
      response =
          FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session ID cannot be null", nullptr));
    } else if (inputs_value == nullptr || fl_value_get_type(inputs_value) != FL_VALUE_TYPE_MAP) {
      ERROR_LOG("Inputs cannot be null or is not a map");
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("NULL_INPUTS", "Inputs cannot be null", nullptr));
    } else {
      const char *session_id = fl_value_get_string(session_id_value);
      DEBUG_LOG("Session ID: " << session_id);

      // Check if session exists
      auto session_it = g_sessions.find(session_id);
      if (session_it == g_sessions.end()) {
        ERROR_LOG("Session not found");
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      } else {
        try {
          // Process inference
          DEBUG_LOG("Processing inference with valid session");

          // Prepare input tensors
          auto &session_info = session_it->second;
          DEBUG_LOG("Session has " << session_info.input_names.size() << " inputs and "
                                   << session_info.output_names.size() << " outputs");

          auto inputs_map = fl_value_to_map(inputs_value);
          DEBUG_LOG("Received " << inputs_map.size() << " input values from Flutter");

          std::vector<Ort::Value> ort_inputs;
          std::vector<const char *> input_names_cstr;
          // store input buffers unless the values will be cleaned after the
          // current iteration
          std::vector<std::vector<float>> input_buffers;

          // For simplicity, we'll just handle float inputs in this example
          for (const auto &name : session_info.input_names) {
            DEBUG_LOG("Processing input: " << name);
            auto it = inputs_map.find(name);

            if (it != inputs_map.end()) {
              DEBUG_LOG("Found input in map: " << name);
              // Check if this is an OrtValue reference
              if (fl_value_get_type(it->second) == FL_VALUE_TYPE_MAP) {
                DEBUG_LOG("Input is an OrtValue reference");
                // Check for valueId in the map
                FlValue *value_id_val = fl_value_lookup_string(it->second, "valueId");
                if (value_id_val != nullptr && fl_value_get_type(value_id_val) == FL_VALUE_TYPE_STRING) {
                  const char *value_id = fl_value_get_string(value_id_val);
                  DEBUG_LOG("OrtValue ID: " << value_id);
                  // Look up the OrtValue in the global map
                  auto ort_value_it = g_ort_values.find(value_id);
                  if (ort_value_it != g_ort_values.end()) {
                    DEBUG_LOG("Found OrtValue in global map");

                    // First check if the tensor exists
                    if (ort_value_it->second && ort_value_it->second->IsTensor()) {
                      DEBUG_LOG("OrtValue is a valid tensor");

                      // Use the actual tensor from the OrtValue instead of a dummy tensor
                      DEBUG_LOG("Using actual OrtValue tensor");

                      // Get tensor info for debugging
                      Ort::TypeInfo type_info = ort_value_it->second->GetTypeInfo();
                      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                      // ONNXTensorElementDataType element_type = tensor_info.GetElementType();
                      auto shape = tensor_info.GetShape();

                      // Log shape for debugging
                      std::ostringstream shape_str;
                      shape_str << "Input tensor shape: [";
                      for (size_t i = 0; i < shape.size(); i++) {
                        shape_str << shape[i];
                        if (i < shape.size() - 1)
                          shape_str << ", ";
                      }
                      shape_str << "]";
                      DEBUG_LOG(shape_str.str());

                      // Move the tensor to ort_inputs
                      // Note: After this operation, the tensor in g_ort_values will be empty
                      // This is fine if the tensor is only used for this inference
                      ort_inputs.push_back(std::move(*ort_value_it->second));
                      input_names_cstr.push_back(name.c_str());
                      continue;
                    } else {
                      // Handle invalid tensor
                      ERROR_LOG("The referenced tensor is not valid");
                      response = FL_METHOD_RESPONSE(fl_method_error_response_new(
                          "INVALID_TENSOR", "The referenced tensor is not valid", nullptr));
                      return;
                    }
                  } else {
                    // OrtValue not found
                    ERROR_LOG("OrtValue with ID " << value_id << " not found");
                    response = FL_METHOD_RESPONSE(fl_method_error_response_new(
                        "INVALID_ORT_VALUE", ("OrtValue with ID " + std::string(value_id) + " not found").c_str(),
                        nullptr));
                    return;
                  }
                }
              }

              // TODO: only accept OrtValue input
              // Handle regular list input if not an OrtValue
              if (fl_value_get_type(it->second) == FL_VALUE_TYPE_LIST) {
                DEBUG_LOG("Input is a regular list");
                // Get shape if provided
                std::vector<int64_t> shape;
                auto shape_it = inputs_map.find(name + "_shape");
                if (shape_it != inputs_map.end() && fl_value_get_type(shape_it->second) == FL_VALUE_TYPE_LIST) {
                  shape = fl_value_to_vector<int64_t>(shape_it->second);
                  DEBUG_LOG("Found custom shape for input");
                } else {
                  // Default shape: just a single dimension with the list length
                  shape.push_back(fl_value_get_length(it->second));
                  DEBUG_LOG("Using default shape with length: " << shape[0]);
                }

                // Log the shape
                std::ostringstream shape_str;
                shape_str << "Shape: [";
                for (size_t i = 0; i < shape.size(); i++) {
                  shape_str << shape[i];
                  if (i < shape.size() - 1)
                    shape_str << ", ";
                }
                shape_str << "]";
                DEBUG_LOG(shape_str.str());

                // Convert to float vector and persist it in the container
                std::vector<float> data = fl_value_to_vector<float>(it->second);
                DEBUG_LOG("Converted data to float vector with " << data.size() << " elements");

                input_buffers.push_back(std::move(data));
                auto &buffer = input_buffers.back();

                // Create tensor
                DEBUG_LOG("Creating tensor from data");
                Ort::MemoryInfo memory_info =
                    Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

                Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, buffer.data(), buffer.size(),
                                                                    shape.data(), shape.size());

                DEBUG_LOG("Adding tensor to ort_inputs");
                ort_inputs.push_back(std::move(tensor));
                input_names_cstr.push_back(name.c_str());
              }
            } else {
              DEBUG_LOG("Warning: Input name " << name << " not found in inputs map");
            }
          }

          // Prepare output names
          std::vector<const char *> output_names_cstr;
          for (const auto &name : session_info.output_names) {
            output_names_cstr.push_back(name.c_str());
          }

          DEBUG_LOG("Prepared " << input_names_cstr.size() << " input tensors");
          DEBUG_LOG("Prepared " << output_names_cstr.size() << " output names");

          // Run inference
          DEBUG_LOG("Running inference with session->Run()");
          auto output_tensors =
              session_info.session->Run(Ort::RunOptions{nullptr}, input_names_cstr.data(), ort_inputs.data(),
                                        ort_inputs.size(), output_names_cstr.data(), output_names_cstr.size());
          DEBUG_LOG("Inference completed successfully");

          // Create the outputs map manually (removed g_autoptr)
          FlValue *result_map = fl_value_new_map();  // ref count = 1
          FlValue *outputs_map = fl_value_new_map(); // ref count = 1

          // Process outputs safely
          DEBUG_LOG("Processing " << output_tensors.size() << " output tensors");
          for (size_t i = 0; i < output_tensors.size(); i++) {
            const auto &output_name = session_info.output_names[i];
            DEBUG_LOG("Processing output: " << output_name);
            auto &tensor = output_tensors[i];

            // Get tensor info
            Ort::TypeInfo type_info = tensor.GetTypeInfo();
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType tensor_type = tensor_info.GetElementType();
            auto shape = tensor_info.GetShape();

            // Log shape
            std::ostringstream shape_str;
            shape_str << "Output shape: [";
            for (size_t i = 0; i < shape.size(); i++) {
              shape_str << shape[i];
              if (i < shape.size() - 1)
                shape_str << ", ";
            }
            shape_str << "]";
            DEBUG_LOG(shape_str.str());

            DEBUG_LOG("Output data type: " << tensor_type);

            FlValue *shape_list = vector_to_fl_value<int64_t>(shape); // shape_list ref=1
            fl_value_set_string(outputs_map, (output_name + "_shape").c_str(), shape_list);
            fl_value_unref(shape_list); // Decrement ref count after adding to map

            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
              DEBUG_LOG("Processing float data");
              // Calculate flat size
              size_t flat_size = 1;
              for (auto dim : shape) {
                flat_size *= (dim > 0) ? dim : 1; // Handle dynamic dimensions
              }
              DEBUG_LOG("Flat size: " << flat_size);

              // Get float data directly from the output tensor in the vector
              const float *data = tensor.GetTensorData<float>();
              if (data != nullptr && tensor.IsTensor()) {
                DEBUG_LOG("Copying data to vector");
                // Copy data into a vector
                std::vector<float> data_vec(data, data + flat_size);

                // Log a few values for debugging
                std::ostringstream values_str;
                values_str << "First few values: ";
                for (size_t j = 0; j < std::min(flat_size, size_t(5)); j++) {
                  values_str << data_vec[j];
                  if (j < std::min(flat_size, size_t(5)) - 1)
                    values_str << ", ";
                }
                DEBUG_LOG(values_str.str());

                FlValue *data_list = vector_to_fl_value<float>(data_vec); // data_list ref=1
                DEBUG_LOG("Setting data in outputs_map");
                fl_value_set_string(outputs_map, output_name.c_str(), data_list);
                fl_value_unref(data_list); // Decrement ref count after adding to map
                DEBUG_LOG("Successfully added float data to outputs map");
              } else {
                ERROR_LOG("Data pointer is null or tensor is invalid");
              }
            } else if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
              DEBUG_LOG("Processing int32 data");
              // Calculate flat size
              size_t flat_size = 1;
              for (auto dim : shape) {
                flat_size *= (dim > 0) ? dim : 1; // Handle dynamic dimensions
              }
              DEBUG_LOG("Flat size: " << flat_size);

              // Get int data directly
              const int32_t *data = tensor.GetTensorData<int32_t>();
              if (data != nullptr) {
                DEBUG_LOG("Copying data to vector");
                std::vector<int> data_vec(data, data + flat_size);

                // Log a few values for debugging
                std::ostringstream values_str;
                values_str << "First few values: ";
                for (size_t j = 0; j < std::min(flat_size, size_t(5)); j++) {
                  values_str << data_vec[j];
                  if (j < std::min(flat_size, size_t(5)) - 1)
                    values_str << ", ";
                }
                DEBUG_LOG(values_str.str());

                FlValue *data_list = vector_to_fl_value<int>(data_vec); // data_list ref=1
                DEBUG_LOG("Setting data in outputs_map");
                fl_value_set_string(outputs_map, output_name.c_str(), data_list);
                fl_value_unref(data_list); // Decrement ref count after adding to map
                DEBUG_LOG("Successfully added int32 data to outputs map");
              } else {
                ERROR_LOG("Data pointer is null");
              }
            }
            // Add more data types as needed
            else {
              DEBUG_LOG("Unsupported tensor type: " << tensor_type);
            }
          }

          // Set the outputs map in the result
          DEBUG_LOG("Adding outputs map to result");
          fl_value_set_string(result_map, "outputs", outputs_map);

          // Return the result
          DEBUG_LOG("Creating success response for runInference");
          response = FL_METHOD_RESPONSE(fl_method_success_response_new(result_map));
          DEBUG_LOG("Success response created");
        } catch (const Ort::Exception &e) {
          ERROR_LOG("ORT Exception: " << e.what());
          response = FL_METHOD_RESPONSE(fl_method_error_response_new("ORT_ERROR", e.what(), nullptr));
        } catch (const std::exception &e) {
          ERROR_LOG("Generic Exception: " << e.what());
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
      ERROR_LOG("getMetadata: Invalid or missing sessionId");
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    const char *session_id = fl_value_get_string(session_id_value);

    // Check if the session exists
    if (g_sessions.find(session_id) == g_sessions.end()) {
      ERROR_LOG("getMetadata: Session not found in g_sessions map, map size: " << g_sessions.size());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    try {
      // Get the session
      auto &session_info = g_sessions[session_id];
      auto &session = session_info.session;

      if (!session) {
        ERROR_LOG("getMetadata: Session pointer is null");
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session is invalid", nullptr));
        return;
      }

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
      ERROR_LOG("getMetadata: ORT_ERROR: " << e.what() << ", error code: " << e.GetOrtErrorCode());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("ORT_ERROR", e.what(), nullptr));
    } catch (const std::exception &e) {
      ERROR_LOG("getMetadata: GENERIC_ERROR: " << e.what());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("GENERIC_ERROR", e.what(), nullptr));
    } catch (...) {
      ERROR_LOG("getMetadata: UNKNOWN_ERROR: Unknown exception occurred");
      response =
          FL_METHOD_RESPONSE(fl_method_error_response_new("UNKNOWN_ERROR", "Unknown exception occurred", nullptr));
    }
  } else if (strcmp(method, "getInputInfo") == 0) {
    // Extract the session ID
    FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

    if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
      ERROR_LOG("getInputInfo: Invalid or missing sessionId");
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    const char *session_id = fl_value_get_string(session_id_value);

    // Check if the session exists
    if (g_sessions.find(session_id) == g_sessions.end()) {
      ERROR_LOG("getInputInfo: Session not found in g_sessions map, map size: " << g_sessions.size());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    try {
      // Get the session
      auto &session_info = g_sessions[session_id];
      auto &session = session_info.session;

      if (!session) {
        ERROR_LOG("getInputInfo: Session pointer is null");
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session is invalid", nullptr));
        return;
      }

      // Create result list
      FlValue *node_info_list = fl_value_new_list();

      // Get all input info
      size_t num_inputs = session->GetInputCount();
      Ort::AllocatorWithDefaultOptions allocator;

      for (size_t i = 0; i < num_inputs; i++) {
        try {
          auto input_name = session->GetInputNameAllocated(i, allocator);
          auto type_info = session->GetInputTypeInfo(i);

          FlValue *info_map = fl_value_new_map();
          fl_value_set_string_take(info_map, "name", fl_value_new_string(input_name.get()));

          // Check if it's a tensor type
          ONNXType onnx_type = type_info.GetONNXType();

          if (onnx_type == ONNX_TYPE_TENSOR) {
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
        } catch (const Ort::Exception &e) {
          ERROR_LOG("getInputInfo: Error processing input " << i << ": " << e.what()
                                                            << ", error code: " << e.GetOrtErrorCode());
          // Continue with next input instead of failing completely
        } catch (const std::exception &e) {
          ERROR_LOG("getInputInfo: Generic error processing input " << i << ": " << e.what());
          // Continue with next input instead of failing completely
        }
      }

      response = FL_METHOD_RESPONSE(fl_method_success_response_new(node_info_list));
    } catch (const Ort::Exception &e) {
      ERROR_LOG("getInputInfo: ORT_ERROR: " << e.what() << ", error code: " << e.GetOrtErrorCode());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("ORT_ERROR", e.what(), nullptr));
    } catch (const std::exception &e) {
      ERROR_LOG("getInputInfo: GENERIC_ERROR: " << e.what());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("GENERIC_ERROR", e.what(), nullptr));
    } catch (...) {
      ERROR_LOG("getInputInfo: UNKNOWN_ERROR: Unknown exception occurred");
      response =
          FL_METHOD_RESPONSE(fl_method_error_response_new("UNKNOWN_ERROR", "Unknown exception occurred", nullptr));
    }
  } else if (strcmp(method, "getOutputInfo") == 0) {
    // Extract the session ID
    FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

    if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
      ERROR_LOG("getOutputInfo: Invalid or missing sessionId");
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    const char *session_id = fl_value_get_string(session_id_value);

    // Check if the session exists
    if (g_sessions.find(session_id) == g_sessions.end()) {
      ERROR_LOG("getOutputInfo: Session not found in g_sessions map, map size: " << g_sessions.size());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session not found", nullptr));
      return;
    }

    try {
      // Get the session
      auto &session_info = g_sessions[session_id];
      auto &session = session_info.session;

      if (!session) {
        ERROR_LOG("getOutputInfo: Session pointer is null");
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SESSION", "Session is invalid", nullptr));
        return;
      }

      // Create result list
      FlValue *node_info_list = fl_value_new_list();

      // Get all output info
      size_t num_outputs = session->GetOutputCount();
      Ort::AllocatorWithDefaultOptions allocator;

      for (size_t i = 0; i < num_outputs; i++) {
        try {
          auto output_name = session->GetOutputNameAllocated(i, allocator);
          auto type_info = session->GetOutputTypeInfo(i);

          FlValue *info_map = fl_value_new_map();
          fl_value_set_string_take(info_map, "name", fl_value_new_string(output_name.get()));

          // Check if it's a tensor type
          ONNXType onnx_type = type_info.GetONNXType();

          if (onnx_type == ONNX_TYPE_TENSOR) {
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
        } catch (const Ort::Exception &e) {
          ERROR_LOG("getOutputInfo: Error processing output " << i << ": " << e.what()
                                                              << ", error code: " << e.GetOrtErrorCode());
          // Continue with next output instead of failing completely
        } catch (const std::exception &e) {
          ERROR_LOG("getOutputInfo: Generic error processing output " << i << ": " << e.what());
          // Continue with next output instead of failing completely
        }
      }

      response = FL_METHOD_RESPONSE(fl_method_success_response_new(node_info_list));
    } catch (const Ort::Exception &e) {
      ERROR_LOG("getOutputInfo: ORT_ERROR: " << e.what() << ", error code: " << e.GetOrtErrorCode());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("ORT_ERROR", e.what(), nullptr));
    } catch (const std::exception &e) {
      ERROR_LOG("getOutputInfo: GENERIC_ERROR: " << e.what());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("GENERIC_ERROR", e.what(), nullptr));
    } catch (...) {
      ERROR_LOG("getOutputInfo: UNKNOWN_ERROR: Unknown exception occurred");
      response =
          FL_METHOD_RESPONSE(fl_method_error_response_new("UNKNOWN_ERROR", "Unknown exception occurred", nullptr));
    }
  } else if (strcmp(method, "createOrtValue") == 0) {
    try {
      // Extract arguments
      FlValue *source_type_value = fl_value_lookup_string(args, "sourceType");
      FlValue *data_value = fl_value_lookup_string(args, "data");
      FlValue *shape_value = fl_value_lookup_string(args, "shape");

      DEBUG_LOG("createOrtValue called with parameters:");

      if (source_type_value == nullptr || data_value == nullptr || shape_value == nullptr) {
        ERROR_LOG("Missing required arguments for createOrtValue");
        response =
            FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_ARGS", "Missing required arguments", nullptr));
      } else {
        const char *source_type = fl_value_get_string(source_type_value);

        DEBUG_LOG("  sourceType: " << source_type);
        DEBUG_LOG("  data type: " << fl_value_type_to_string(fl_value_get_type(data_value)));

        // Get shape from argument
        std::vector<int64_t> shape_vec = fl_value_to_vector<int64_t>(shape_value);
        if (shape_vec.empty()) {
          ERROR_LOG("Shape vector is empty");
          response =
              FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_SHAPE", "Shape cannot be empty", nullptr));
        } else {
          // Log the shape
          std::ostringstream shape_str;
          shape_str << "  shape: [";
          for (size_t i = 0; i < shape_vec.size(); i++) {
            shape_str << shape_vec[i];
            if (i < shape_vec.size() - 1)
              shape_str << ", ";
          }
          shape_str << "]";
          DEBUG_LOG(shape_str.str());

          // Create tensor based on source data type
          std::string error_message;

          if (strcmp(source_type, "float32") == 0) {
            // Check for both standard lists and typed data
            FlValueType data_type = fl_value_get_type(data_value);
            int data_type_val = static_cast<int>(data_type);
            DEBUG_LOG("Float32 data received with type: " << fl_value_type_to_string(data_type));
            DEBUG_LOG("Data type value: " << data_type_val);

            std::vector<float> float_data;
            if (data_type == FL_VALUE_TYPE_FLOAT32_LIST) {
              size_t length = fl_value_get_length(data_value);
              DEBUG_LOG("Float32List with length: " << length);
              float_data.reserve(length);

              const float *float_array = fl_value_get_float32_list(data_value);
              float_data.assign(float_array, float_array + length);

              // Note: the fl_value_get_float() does not work for Float32List
            } else {
              ERROR_LOG("Expected Float32List, got " << fl_value_type_to_string(data_type));
            }

            DEBUG_LOG("Float data size: " << float_data.size());
            // Verify data size matches shape
            size_t total_elements = 1;
            for (auto dim : shape_vec) {
              total_elements *= dim;
            }

            if (float_data.size() != total_elements) {
              ERROR_LOG("Data size (" << float_data.size() << ") doesn't match shape (" << total_elements
                                      << " elements)");
              response = FL_METHOD_RESPONSE(
                  fl_method_error_response_new("SHAPE_MISMATCH", "Data size doesn't match provided shape", nullptr));
              return;
            }

            // Create ORT tensor
            Ort::MemoryInfo memory_info =
                Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

            // tensor will be reshaped from a flatten list of floats
            Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, float_data.data(), float_data.size(),
                                                                shape_vec.data(), shape_vec.size());

            // Generate and store value ID
            // std::string value_id = "tensor_" + generate_session_uuid();
            std::string value_id = generate_ort_value_uuid();
            g_ort_values[value_id] = std::make_unique<Ort::Value>(std::move(tensor));

            // set device to cpu
            const char *device = "cpu";

            // Create response
            FlValue *result_map = fl_value_new_map();
            fl_value_set_string_take(result_map, "valueId", fl_value_new_string(value_id.c_str()));
            fl_value_set_string_take(result_map, "dataType", fl_value_new_string("float32"));
            fl_value_set_string_take(result_map, "shape", vector_to_fl_value<int64_t>(shape_vec));
            fl_value_set_string_take(result_map, "device", fl_value_new_string(device));

            response = FL_METHOD_RESPONSE(fl_method_success_response_new(result_map));
          } else if (strcmp(source_type, "int32") == 0) {
            DEBUG_LOG("int32");
          } else if (strcmp(source_type, "int64") == 0) {
            DEBUG_LOG("int64");
          } else if (strcmp(source_type, "float16") == 0) {
            DEBUG_LOG("float16");
          } else {
            ERROR_LOG("Unsupported data type: " << source_type);
            response = FL_METHOD_RESPONSE(fl_method_error_response_new(
                "UNSUPPORTED_TYPE", ("Unsupported data type: " + std::string(source_type)).c_str(), nullptr));
          }
        }
      }
    } catch (const std::exception &e) {
      ERROR_LOG("Exception in createOrtValue: " << e.what());
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("TENSOR_CREATION_ERROR", e.what(), nullptr));
    }
  } else if (strcmp(method, "convertOrtValue") == 0) {
    try {
      // Extract arguments
      FlValue *value_id_value = fl_value_lookup_string(args, "valueId");
      FlValue *target_type_value = fl_value_lookup_string(args, "targetType");

      if (value_id_value == nullptr || target_type_value == nullptr) {
        response =
            FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_ARGS", "Missing required arguments", nullptr));
        return;
      }

      const char *value_id = fl_value_get_string(value_id_value);
      const char *target_type = fl_value_get_string(target_type_value);

      // Call C function to convert tensor
      char *error_ptr = nullptr;
      char *result_json = ort_convert_tensor(value_id, target_type, &error_ptr);

      if (result_json == nullptr) {
        if (error_ptr != nullptr) {
          response = FL_METHOD_RESPONSE(fl_method_error_response_new("CONVERSION_ERROR", error_ptr, nullptr));
          free(error_ptr);
        } else {
          response =
              FL_METHOD_RESPONSE(fl_method_error_response_new("CONVERSION_ERROR", "Failed to convert tensor", nullptr));
        }
      } else {
        // Parse JSON result into FlValue
        std::string result_str(result_json);
        free(result_json); // Free the C string

        // Create a map to return
        FlValue *result_map = fl_value_new_map();

        // Extract valueId from JSON
        size_t id_pos = result_str.find("\"valueId\":\"") + 11;
        size_t id_end = result_str.find("\"", id_pos);
        std::string new_value_id = result_str.substr(id_pos, id_end - id_pos);

        fl_value_set_string_take(result_map, "valueId", fl_value_new_string(new_value_id.c_str()));

        // Extract dataType from JSON
        size_t type_pos = result_str.find("\"dataType\":\"") + 12;
        size_t type_end = result_str.find("\"", type_pos);
        std::string data_type = result_str.substr(type_pos, type_end - type_pos);

        fl_value_set_string_take(result_map, "dataType", fl_value_new_string(data_type.c_str()));

        // Extract shape from JSON and convert to vector
        size_t shape_pos = result_str.find("\"shape\":[") + 9;
        size_t shape_end = result_str.find("]", shape_pos);
        std::string shape_str = result_str.substr(shape_pos, shape_end - shape_pos);

        // Parse shape string into vector
        std::vector<int64_t> shape_vec;
        std::istringstream shape_stream(shape_str);
        std::string dim;
        while (std::getline(shape_stream, dim, ',')) {
          shape_vec.push_back(std::stoll(dim));
        }

        fl_value_set_string_take(result_map, "shape", vector_to_fl_value<int64_t>(shape_vec));

        // Extract device from JSON
        size_t device_pos = result_str.find("\"device\":\"") + 10;
        size_t device_end = result_str.find("\"", device_pos);
        std::string device = result_str.substr(device_pos, device_end - device_pos);

        fl_value_set_string_take(result_map, "device", fl_value_new_string(device.c_str()));

        // Create success response
        response = FL_METHOD_RESPONSE(fl_method_success_response_new(result_map));
      }
    } catch (const std::exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("CONVERSION_ERROR", e.what(), nullptr));
    }
  } else if (strcmp(method, "moveOrtValueToDevice") == 0) {
    try {
      // Extract arguments
      FlValue *value_id_value = fl_value_lookup_string(args, "valueId");
      FlValue *target_device_value = fl_value_lookup_string(args, "targetDevice");

      if (value_id_value == nullptr || target_device_value == nullptr) {
        response =
            FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_ARGS", "Missing required arguments", nullptr));
        return;
      }

      const char *value_id = fl_value_get_string(value_id_value);
      const char *target_device = fl_value_get_string(target_device_value);

      // Call C function to move tensor
      char *error_ptr = nullptr;
      char *result_json = ort_move_tensor_to_device(value_id, target_device, &error_ptr);

      if (result_json == nullptr) {
        if (error_ptr != nullptr) {
          response = FL_METHOD_RESPONSE(fl_method_error_response_new("DEVICE_TRANSFER_ERROR", error_ptr, nullptr));
          free(error_ptr);
        } else {
          response = FL_METHOD_RESPONSE(
              fl_method_error_response_new("DEVICE_TRANSFER_ERROR", "Failed to move tensor to device", nullptr));
        }
      } else {
        // Parse JSON result into FlValue
        std::string result_str(result_json);
        free(result_json); // Free the C string

        // Create a map to return
        FlValue *result_map = fl_value_new_map();

        // Extract valueId from JSON
        size_t id_pos = result_str.find("\"valueId\":\"") + 11;
        size_t id_end = result_str.find("\"", id_pos);
        std::string new_value_id = result_str.substr(id_pos, id_end - id_pos);

        fl_value_set_string_take(result_map, "valueId", fl_value_new_string(new_value_id.c_str()));

        // Extract dataType from JSON
        size_t type_pos = result_str.find("\"dataType\":\"") + 12;
        size_t type_end = result_str.find("\"", type_pos);
        std::string data_type = result_str.substr(type_pos, type_end - type_pos);

        fl_value_set_string_take(result_map, "dataType", fl_value_new_string(data_type.c_str()));

        // Extract shape from JSON and convert to vector
        size_t shape_pos = result_str.find("\"shape\":[") + 9;
        size_t shape_end = result_str.find("]", shape_pos);
        std::string shape_str = result_str.substr(shape_pos, shape_end - shape_pos);

        // Parse shape string into vector
        std::vector<int64_t> shape_vec;
        std::istringstream shape_stream(shape_str);
        std::string dim;
        while (std::getline(shape_stream, dim, ',')) {
          shape_vec.push_back(std::stoll(dim));
        }

        fl_value_set_string_take(result_map, "shape", vector_to_fl_value<int64_t>(shape_vec));

        // Extract device from JSON
        size_t device_pos = result_str.find("\"device\":\"") + 10;
        size_t device_end = result_str.find("\"", device_pos);
        std::string device = result_str.substr(device_pos, device_end - device_pos);

        fl_value_set_string_take(result_map, "device", fl_value_new_string(device.c_str()));

        // Create success response
        response = FL_METHOD_RESPONSE(fl_method_success_response_new(result_map));
      }
    } catch (const std::exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("DEVICE_TRANSFER_ERROR", e.what(), nullptr));
    }
  } else if (strcmp(method, "getOrtValueData") == 0) {
    try {
      // Extract arguments
      FlValue *value_id_value = fl_value_lookup_string(args, "valueId");
      FlValue *data_type_value = fl_value_lookup_string(args, "dataType");

      if (value_id_value == nullptr) {
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_ARGS", "Missing value ID", nullptr));
        return;
      }

      const char *value_id = fl_value_get_string(value_id_value);
      const char *data_type = (data_type_value != nullptr) ? fl_value_get_string(data_type_value) : nullptr;

      // Call C function to get tensor data
      char *error_ptr = nullptr;
      char *result_json = ort_get_tensor_data(value_id, data_type, &error_ptr);

      if (result_json == nullptr) {
        if (error_ptr != nullptr) {
          response = FL_METHOD_RESPONSE(fl_method_error_response_new("DATA_EXTRACTION_ERROR", error_ptr, nullptr));
          free(error_ptr);
        } else {
          response = FL_METHOD_RESPONSE(
              fl_method_error_response_new("DATA_EXTRACTION_ERROR", "Failed to extract tensor data", nullptr));
        }
      } else {
        // Parse JSON result - this is a simplified approach
        std::string result_str(result_json);
        free(result_json); // Free the C string

        // Create a map to return
        FlValue *result_map = fl_value_new_map();

        // For actual implementation, you would need a proper JSON parser
        // This simplified approach works for the expected format but is not robust

        // Extract shape from JSON - simplified parsing
        size_t shape_pos = result_str.find("\"shape\":[") + 9;
        size_t shape_end = result_str.find("]", shape_pos);
        std::string shape_str = result_str.substr(shape_pos, shape_end - shape_pos);

        // Parse shape string into vector
        std::vector<int64_t> shape_vec;
        std::istringstream shape_stream(shape_str);
        std::string dim;
        while (std::getline(shape_stream, dim, ',')) {
          if (!dim.empty()) {
            shape_vec.push_back(std::stoll(dim));
          }
        }

        fl_value_set_string_take(result_map, "shape", vector_to_fl_value<int64_t>(shape_vec));

        // Extract data from JSON - simplified parsing
        size_t data_pos = result_str.find("\"data\":[") + 8;
        size_t data_end = result_str.find("]", data_pos);
        std::string data_str = result_str.substr(data_pos, data_end - data_pos);

        // Parse data string into appropriate type based on data_type
        if (data_type == nullptr || strcmp(data_type, "float32") == 0) {
          // Parse as float
          std::vector<float> data_vec;
          std::istringstream data_stream(data_str);
          std::string value;
          while (std::getline(data_stream, value, ',')) {
            if (!value.empty()) {
              data_vec.push_back(std::stof(value));
            }
          }

          fl_value_set_string_take(result_map, "data", vector_to_fl_value<float>(data_vec));
        } else if (strcmp(data_type, "int32") == 0) {
          // Parse as int
          std::vector<int> data_vec;
          std::istringstream data_stream(data_str);
          std::string value;
          while (std::getline(data_stream, value, ',')) {
            if (!value.empty()) {
              data_vec.push_back(std::stoi(value));
            }
          }

          fl_value_set_string_take(result_map, "data", vector_to_fl_value<int>(data_vec));
        } else {
          // Default to float
          std::vector<float> data_vec;
          std::istringstream data_stream(data_str);
          std::string value;
          while (std::getline(data_stream, value, ',')) {
            if (!value.empty()) {
              data_vec.push_back(std::stof(value));
            }
          }

          fl_value_set_string_take(result_map, "data", vector_to_fl_value<float>(data_vec));
        }

        // Create success response
        response = FL_METHOD_RESPONSE(fl_method_success_response_new(result_map));
      }
    } catch (const std::exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("DATA_EXTRACTION_ERROR", e.what(), nullptr));
    }
  } else if (strcmp(method, "releaseOrtValue") == 0) {
    try {
      // Extract arguments
      FlValue *value_id_value = fl_value_lookup_string(args, "valueId");

      if (value_id_value == nullptr) {
        response = FL_METHOD_RESPONSE(fl_method_error_response_new("INVALID_ARGS", "Missing value ID", nullptr));
        return;
      }

      const char *value_id = fl_value_get_string(value_id_value);

      // Call C function to release tensor
      char *error_ptr = nullptr;
      bool success = ort_release_tensor(value_id, &error_ptr);

      if (!success) {
        if (error_ptr != nullptr) {
          response = FL_METHOD_RESPONSE(fl_method_error_response_new("RELEASE_ERROR", error_ptr, nullptr));
          free(error_ptr);
        } else {
          response =
              FL_METHOD_RESPONSE(fl_method_error_response_new("RELEASE_ERROR", "Failed to release tensor", nullptr));
        }
      } else {
        // Create success response
        response = FL_METHOD_RESPONSE(fl_method_success_response_new(fl_value_new_null()));
      }
    } catch (const std::exception &e) {
      response = FL_METHOD_RESPONSE(fl_method_error_response_new("RELEASE_ERROR", e.what(), nullptr));
    }
  } else {
    response = FL_METHOD_RESPONSE(fl_method_not_implemented_response_new());
  }

  if (response != nullptr) {
    DEBUG_LOG("Sending response for method: " << method);
    fl_method_call_respond(method_call, response, nullptr);
    DEBUG_LOG("Response sent for method: " << method);
  } else {
    ERROR_LOG("No response created for method: " << method << " - this is a critical error!");
    // Create a default error response to avoid FlBinaryMessengerResponseHandle errors
    response = FL_METHOD_RESPONSE(
        fl_method_error_response_new("MISSING_RESPONSE", "No response was created for this method call", nullptr));
    fl_method_call_respond(method_call, response, nullptr);
  }
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
  // Response is now handled inside flutter_onnxruntime_plugin_handle_method_call
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
