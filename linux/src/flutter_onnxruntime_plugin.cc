#include "include/flutter_onnxruntime/flutter_onnxruntime_plugin.h"

#include <flutter_linux/flutter_linux.h>
#include <gtk/gtk.h>
#include <sys/utsname.h>

#include "session_manager.h"
#include "tensor_manager.h"
#include "value_conversion.h"
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>

#define FLUTTER_ONNXRUNTIME_PLUGIN(obj)                                                                                \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), flutter_onnxruntime_plugin_get_type(), FlutterOnnxruntimePlugin))

struct _FlutterOnnxruntimePlugin {
  GObject parent_instance;

  // SessionManager for handling ONNX Runtime sessions
  SessionManager *session_manager;

  // TensorManager for handling OrtValue objects
  TensorManager *tensor_manager;

  // Maps to store value data
  std::map<std::string, void *> values;

  // Mutex for thread safety
  std::mutex mutex;
};

G_DEFINE_TYPE(FlutterOnnxruntimePlugin, flutter_onnxruntime_plugin, g_object_get_type())

// Method declarations
static void flutter_onnxruntime_plugin_dispose(GObject *object);
static void flutter_onnxruntime_plugin_handle_method_call(FlutterOnnxruntimePlugin *self, FlMethodCall *method_call);
static void method_call_handler(FlMethodChannel *channel, FlMethodCall *method_call, gpointer user_data);

// Helper function to get platform version
static FlValue *get_platform_version();

// Session management
static FlValue *create_session(FlutterOnnxruntimePlugin *self, FlValue *args);
static FlValue *run_inference(FlutterOnnxruntimePlugin *self, FlValue *args);
static FlValue *close_session(FlutterOnnxruntimePlugin *self, FlValue *args);
static FlValue *get_metadata(FlutterOnnxruntimePlugin *self, FlValue *args);
static FlValue *get_input_info(FlutterOnnxruntimePlugin *self, FlValue *args);
static FlValue *get_output_info(FlutterOnnxruntimePlugin *self, FlValue *args);

// OrtValue operations
static FlValue *create_ort_value(FlutterOnnxruntimePlugin *self, FlValue *args);
static FlValue *convert_ort_value(FlutterOnnxruntimePlugin *self, FlValue *args);
static FlValue *move_ort_value_to_device(FlutterOnnxruntimePlugin *self, FlValue *args);
static FlValue *get_ort_value_data(FlutterOnnxruntimePlugin *self, FlValue *args);
static FlValue *release_ort_value(FlutterOnnxruntimePlugin *self, FlValue *args);

// Plugin class initialization
static void flutter_onnxruntime_plugin_class_init(FlutterOnnxruntimePluginClass *klass) {
  G_OBJECT_CLASS(klass)->dispose = flutter_onnxruntime_plugin_dispose;
}

static void flutter_onnxruntime_plugin_init(FlutterOnnxruntimePlugin *self) {
  self->session_manager = new SessionManager();
  self->tensor_manager = new TensorManager();
}

static void flutter_onnxruntime_plugin_dispose(GObject *object) {
  FlutterOnnxruntimePlugin *self = FLUTTER_ONNXRUNTIME_PLUGIN(object);

  // Clean up session manager, tensor manager and values
  delete self->session_manager;
  delete self->tensor_manager;

  std::lock_guard<std::mutex> lock(self->mutex);
  self->values.clear();

  G_OBJECT_CLASS(flutter_onnxruntime_plugin_parent_class)->dispose(object);
}

// Plugin registration with Flutter engine
void flutter_onnxruntime_plugin_register_with_registrar(FlPluginRegistrar *registrar) {
  FlutterOnnxruntimePlugin *plugin =
      FLUTTER_ONNXRUNTIME_PLUGIN(g_object_new(flutter_onnxruntime_plugin_get_type(), nullptr));

  g_autoptr(FlStandardMethodCodec) codec = fl_standard_method_codec_new();
  g_autoptr(FlMethodChannel) channel = fl_method_channel_new(fl_plugin_registrar_get_messenger(registrar),
                                                             "flutter_onnxruntime", FL_METHOD_CODEC(codec));

  // Setup method call handler
  fl_method_channel_set_method_call_handler(channel, method_call_handler, g_object_ref(plugin), g_object_unref);

  g_object_unref(plugin);
}

// Method call handler function
static void method_call_handler(FlMethodChannel *channel, FlMethodCall *method_call, gpointer user_data) {
  FlutterOnnxruntimePlugin *self = FLUTTER_ONNXRUNTIME_PLUGIN(user_data);
  flutter_onnxruntime_plugin_handle_method_call(self, method_call);
}

// Handler for method calls from Dart
static void flutter_onnxruntime_plugin_handle_method_call(FlutterOnnxruntimePlugin *self, FlMethodCall *method_call) {
  g_autoptr(FlMethodResponse) response = nullptr;
  const gchar *method = fl_method_call_get_name(method_call);
  FlValue *args = fl_method_call_get_args(method_call);

  if (strcmp(method, "getPlatformVersion") == 0) {
    FlValue *result = get_platform_version();
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "createSession") == 0) {
    FlValue *result = create_session(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "runInference") == 0) {
    FlValue *result = run_inference(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "closeSession") == 0) {
    FlValue *result = close_session(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "getMetadata") == 0) {
    FlValue *result = get_metadata(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "getInputInfo") == 0) {
    FlValue *result = get_input_info(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "getOutputInfo") == 0) {
    FlValue *result = get_output_info(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "createOrtValue") == 0) {
    FlValue *result = create_ort_value(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "convertOrtValue") == 0) {
    FlValue *result = convert_ort_value(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "moveOrtValueToDevice") == 0) {
    FlValue *result = move_ort_value_to_device(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "getOrtValueData") == 0) {
    FlValue *result = get_ort_value_data(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else if (strcmp(method, "releaseOrtValue") == 0) {
    FlValue *result = release_ort_value(self, args);
    response = FL_METHOD_RESPONSE(fl_method_success_response_new(result));
    fl_value_unref(result);
  } else {
    response = FL_METHOD_RESPONSE(fl_method_not_implemented_response_new());
  }

  fl_method_call_respond(method_call, response, nullptr);
}

// Implementation of method functions
static FlValue *get_platform_version() {
  struct utsname uname_data = {};
  uname(&uname_data);
  return fl_value_new_string(uname_data.version);
}

static FlValue *create_session(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Extract arguments
  FlValue *model_path_value = fl_value_lookup_string(args, "modelPath");

  // Check if model path is provided and valid
  if (model_path_value == nullptr || fl_value_get_type(model_path_value) != FL_VALUE_TYPE_STRING) {
    g_autoptr(FlValue) error_details = fl_value_new_map();
    fl_value_set_string_take(error_details, "message", fl_value_new_string("Model path must be a string"));
    return fl_value_new_map(); // Return empty map for now since we can't throw exceptions here
  }

  const char *model_path = fl_value_get_string(model_path_value);

  // Extract session options if provided
  FlValue *session_options_value = fl_value_lookup_string(args, "sessionOptions");

  // Create session options
  Ort::SessionOptions session_options;

  // Configure session options if provided
  if (session_options_value != nullptr && fl_value_get_type(session_options_value) == FL_VALUE_TYPE_MAP) {
    auto options_map = fl_value_to_map(session_options_value);

    // Set threading options
    auto intra_threads_val = options_map.find("intraOpNumThreads");
    if (intra_threads_val != options_map.end() && fl_value_get_type(intra_threads_val->second) == FL_VALUE_TYPE_INT) {
      session_options.SetIntraOpNumThreads(fl_value_get_int(intra_threads_val->second));
    }

    auto inter_threads_val = options_map.find("interOpNumThreads");
    if (inter_threads_val != options_map.end() && fl_value_get_type(inter_threads_val->second) == FL_VALUE_TYPE_INT) {
      session_options.SetInterOpNumThreads(fl_value_get_int(inter_threads_val->second));
    }
  }

  // Create session using session manager
  std::string session_id = self->session_manager->createSession(model_path, session_options);

  // Check if session creation failed
  if (session_id.empty()) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Failed to create session"));
    return fl_value_ref(result);
  }

  // Get input and output names
  std::vector<std::string> input_names = self->session_manager->getInputNames(session_id);
  std::vector<std::string> output_names = self->session_manager->getOutputNames(session_id);

  // Create response
  g_autoptr(FlValue) result = fl_value_new_map();
  fl_value_set_string_take(result, "sessionId", fl_value_new_string(session_id.c_str()));

  // Add input names
  FlValue *input_names_value = vector_to_fl_value(input_names);
  fl_value_set_string_take(result, "inputNames", input_names_value);

  // Add output names
  FlValue *output_names_value = vector_to_fl_value(output_names);
  fl_value_set_string_take(result, "outputNames", output_names_value);

  // Status is success
  fl_value_set_string_take(result, "status", fl_value_new_string("success"));

  return fl_value_ref(result);
}

static FlValue *run_inference(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Extract the session ID
  FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");
  if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Invalid session ID"));
    return fl_value_ref(result);
  }
  const char *session_id = fl_value_get_string(session_id_value);

  // Extract the inputs
  FlValue *inputs_value = fl_value_lookup_string(args, "inputs");
  if (inputs_value == nullptr || fl_value_get_type(inputs_value) != FL_VALUE_TYPE_MAP) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Invalid inputs"));
    return fl_value_ref(result);
  }

  // Extract run options if provided
  FlValue *run_options_value = fl_value_lookup_string(args, "runOptions");

  // Get the session
  Ort::Session *session = self->session_manager->getSession(session_id);
  if (session == nullptr) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Session not found"));
    return fl_value_ref(result);
  }

  try {
    // Get input and output names
    std::vector<std::string> input_names = self->session_manager->getInputNames(session_id);
    std::vector<std::string> output_names = self->session_manager->getOutputNames(session_id);

    // Prepare input tensors
    // Note: session.Run() requires a vector of Ort::Value objects so we need to clone the tensors
    // Create a vector to hold the input tensors
    std::vector<Ort::Value> input_tensors;
    std::vector<const char *> input_names_char;

    // get input names from session manager
    for (const auto &name : input_names) {
      input_names_char.push_back(name.c_str());
    }

    // Create memory info for tensor creation
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Iterate through each input
    size_t num_inputs = fl_value_get_length(inputs_value);
    for (size_t i = 0; i < num_inputs; i++) {
      FlValue *key = fl_value_get_map_key(inputs_value, i);
      FlValue *value = fl_value_get_map_value(inputs_value, i);

      if (fl_value_get_type(key) != FL_VALUE_TYPE_STRING || fl_value_get_type(value) != FL_VALUE_TYPE_MAP) {
        continue;
      }

      FlValue *tensor_id_map = fl_value_lookup_string(value, "valueId");

      if (tensor_id_map == nullptr || fl_value_get_type(tensor_id_map) != FL_VALUE_TYPE_STRING) {
        continue;
      }

      std::string tensor_id = fl_value_get_string(tensor_id_map);

      // Get the tensor value
      Ort::Value *tensor_ptr = self->tensor_manager->getTensor(tensor_id);
      if (tensor_ptr != nullptr) {
        // Get tensor info
        Ort::TensorTypeAndShapeInfo tensor_info = tensor_ptr->GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType element_type = tensor_info.GetElementType();
        std::vector<int64_t> shape = tensor_info.GetShape();
        size_t element_count = tensor_info.GetElementCount();

        // Create a new tensor with the same data as the original
        switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
          float *data = tensor_ptr->GetTensorMutableData<float>();
          float *new_data = new float[element_count];
          std::memcpy(new_data, data, element_count * sizeof(float));

          Ort::Value new_tensor =
              Ort::Value::CreateTensor<float>(memory_info, new_data, element_count, shape.data(), shape.size());

          input_tensors.push_back(std::move(new_tensor));
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
          int32_t *data = tensor_ptr->GetTensorMutableData<int32_t>();
          int32_t *new_data = new int32_t[element_count];
          std::memcpy(new_data, data, element_count * sizeof(int32_t));

          Ort::Value new_tensor =
              Ort::Value::CreateTensor<int32_t>(memory_info, new_data, element_count, shape.data(), shape.size());

          input_tensors.push_back(std::move(new_tensor));
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
          int64_t *data = tensor_ptr->GetTensorMutableData<int64_t>();
          int64_t *new_data = new int64_t[element_count];
          std::memcpy(new_data, data, element_count * sizeof(int64_t));

          Ort::Value new_tensor =
              Ort::Value::CreateTensor<int64_t>(memory_info, new_data, element_count, shape.data(), shape.size());

          input_tensors.push_back(std::move(new_tensor));
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
          uint8_t *data = tensor_ptr->GetTensorMutableData<uint8_t>();
          uint8_t *new_data = new uint8_t[element_count];
          std::memcpy(new_data, data, element_count * sizeof(uint8_t));

          Ort::Value new_tensor =
              Ort::Value::CreateTensor<uint8_t>(memory_info, new_data, element_count, shape.data(), shape.size());

          input_tensors.push_back(std::move(new_tensor));
          break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
          bool *data = tensor_ptr->GetTensorMutableData<bool>();
          bool *new_data = new bool[element_count];
          std::memcpy(new_data, data, element_count * sizeof(bool));

          Ort::Value new_tensor =
              Ort::Value::CreateTensor<bool>(memory_info, new_data, element_count, shape.data(), shape.size());

          input_tensors.push_back(std::move(new_tensor));
          break;
        }
        // Add cases for other data types as needed
        default:
          // Skip unsupported tensor types
          break;
        }
      }
    }

    // Prepare output names
    std::vector<const char *> output_names_char;
    for (const auto &name : output_names) {
      output_names_char.push_back(name.c_str());
    }

    // Create and configure run options
    Ort::RunOptions run_options;

    // Configure run options if provided
    if (run_options_value != nullptr && fl_value_get_type(run_options_value) == FL_VALUE_TYPE_MAP) {
      // Extract log severity level if provided
      FlValue *log_severity_level_value = fl_value_lookup_string(run_options_value, "logSeverityLevel");
      if (log_severity_level_value != nullptr && fl_value_get_type(log_severity_level_value) == FL_VALUE_TYPE_INT) {
        run_options.SetRunLogSeverityLevel(fl_value_get_int(log_severity_level_value));
      }

      // Extract log verbosity level if provided
      FlValue *log_verbosity_level_value = fl_value_lookup_string(run_options_value, "logVerbosityLevel");
      if (log_verbosity_level_value != nullptr && fl_value_get_type(log_verbosity_level_value) == FL_VALUE_TYPE_INT) {
        run_options.SetRunLogVerbosityLevel(fl_value_get_int(log_verbosity_level_value));
      }

      // Extract terminate option if provided
      FlValue *terminate_value = fl_value_lookup_string(run_options_value, "terminate");
      if (terminate_value != nullptr && fl_value_get_type(terminate_value) == FL_VALUE_TYPE_BOOL) {
        if (fl_value_get_bool(terminate_value)) {
          run_options.SetTerminate();
        }
      }

      // Add more run options as needed
    }

    // Run inference - only if we have inputs
    std::vector<Ort::Value> output_tensors;

    if (!input_tensors.empty()) {
      output_tensors = session->Run(run_options, input_names_char.data(), input_tensors.data(), input_tensors.size(),
                                    output_names_char.data(), output_names_char.size());
    }

    // Process outputs
    g_autoptr(FlValue) outputs_map = fl_value_new_map();

    // For each output tensor, directly store it using TensorManager's storeTensor
    for (size_t i = 0; i < output_tensors.size(); i++) {
      // Create a tensor ID
      std::string value_id = self->tensor_manager->generateTensorId();

      // Get tensor info before moving
      //   Ort::TensorTypeAndShapeInfo tensor_info = output_tensors[i].GetTensorTypeAndShapeInfo();
      //   std::vector<int64_t> shape = tensor_info.GetShape();

      // Store the tensor directly using storeTensor - this transfers ownership
      self->tensor_manager->storeTensor(value_id, std::move(output_tensors[i]));

      // get the tensor type and shape from tensor manager
      // Note: only do this after storeTensor get the tensor registered in tensor manager
      std::string tensor_type = self->tensor_manager->getTensorType(value_id);
      std::vector<int64_t> shape = self->tensor_manager->getTensorShape(value_id);

      // Add the value ID to the outputs map
      FlValue *shape_list = fl_value_new_list();
      for (const auto &dim : shape) {
        fl_value_append_take(shape_list, fl_value_new_int(dim));
      }

      // Note: Flutter does not allow return a nested map, so we have to use list here to keep the output_info format
      FlValue *output_info = fl_value_new_list();
      fl_value_append_take(output_info, fl_value_new_string(value_id.c_str()));
      fl_value_append_take(output_info, fl_value_new_string(tensor_type.c_str()));
      fl_value_append_take(output_info, shape_list);

      fl_value_set_string_take(outputs_map, output_names[i].c_str(), output_info);
    }

    // Create result
    return fl_value_ref(outputs_map);
  } catch (const Ort::Exception &e) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string(e.what()));
    return fl_value_ref(result);
  } catch (const std::exception &e) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string(e.what()));
    return fl_value_ref(result);
  }
}

static FlValue *close_session(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Get session ID
  FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

  if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
    return fl_value_new_null();
  }

  const char *session_id = fl_value_get_string(session_id_value);

  // Close the session
  self->session_manager->closeSession(session_id);

  return fl_value_new_null();
}

static FlValue *get_metadata(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Extract the session ID
  FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

  if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Invalid session ID"));
    return fl_value_ref(result);
  }

  const char *session_id = fl_value_get_string(session_id_value);

  // Check if the session exists
  if (!self->session_manager->hasSession(session_id)) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Session not found"));
    return fl_value_ref(result);
  }

  try {
    // Get the session
    Ort::Session *session = self->session_manager->getSession(session_id);
    if (!session) {
      g_autoptr(FlValue) result = fl_value_new_map();
      fl_value_set_string_take(result, "error", fl_value_new_string("Session is invalid"));
      return fl_value_ref(result);
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

    // Create empty custom metadata map
    g_autoptr(FlValue) custom_metadata_map = fl_value_new_map();

    // Create response
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "producerName", fl_value_new_string(producer_name.get()));
    fl_value_set_string_take(result, "graphName", fl_value_new_string(graph_name.get()));
    fl_value_set_string_take(result, "domain", fl_value_new_string(domain.get()));
    fl_value_set_string_take(result, "description", fl_value_new_string(description.get()));
    fl_value_set_string_take(result, "version", fl_value_new_int(version));
    fl_value_set_string(result, "customMetadataMap", custom_metadata_map);

    return fl_value_ref(result);
  } catch (const Ort::Exception &e) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string(e.what()));
    return fl_value_ref(result);
  } catch (const std::exception &e) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string(e.what()));
    return fl_value_ref(result);
  }
}

static FlValue *get_input_info(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Extract the session ID
  FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

  if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Invalid session ID"));
    return fl_value_ref(result);
  }

  const char *session_id = fl_value_get_string(session_id_value);

  // Check if the session exists
  if (!self->session_manager->hasSession(session_id)) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Session not found"));
    return fl_value_ref(result);
  }

  try {
    // Get the session
    Ort::Session *session = self->session_manager->getSession(session_id);
    if (!session) {
      g_autoptr(FlValue) result = fl_value_new_map();
      fl_value_set_string_take(result, "error", fl_value_new_string("Session is invalid"));
      return fl_value_ref(result);
    }

    // Get all input info
    size_t num_inputs = session->GetInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    g_autoptr(FlValue) result = fl_value_new_list();

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

          // Convert element type to string
          const char *type_str = "unknown";
          switch (element_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            type_str = "float32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            type_str = "uint8";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            type_str = "int8";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            type_str = "uint16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            type_str = "int16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            type_str = "int32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            type_str = "int64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            type_str = "string";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            type_str = "bool";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            type_str = "float16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            type_str = "double";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            type_str = "uint32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            type_str = "uint64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            type_str = "complex64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            type_str = "complex128";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            type_str = "bfloat16";
            break;
          default:
            type_str = "unknown";
            break;
          }

          fl_value_set_string_take(info_map, "type", fl_value_new_string(type_str));
        } else {
          // Non-tensor type
          fl_value_set_string_take(info_map, "type", fl_value_new_string("non-tensor"));

          // Empty shape for non-tensor types
          FlValue *shape_list = fl_value_new_list();
          fl_value_set_string_take(info_map, "shape", shape_list);
        }

        fl_value_append_take(result, info_map);
      } catch (const Ort::Exception &e) {
        // Skip this input if there's an error
        continue;
      }
    }

    return fl_value_ref(result);
  } catch (const Ort::Exception &e) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string(e.what()));
    return fl_value_ref(result);
  } catch (const std::exception &e) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string(e.what()));
    return fl_value_ref(result);
  }
}

static FlValue *get_output_info(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Extract the session ID
  FlValue *session_id_value = fl_value_lookup_string(args, "sessionId");

  if (session_id_value == nullptr || fl_value_get_type(session_id_value) != FL_VALUE_TYPE_STRING) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Invalid session ID"));
    return fl_value_ref(result);
  }

  const char *session_id = fl_value_get_string(session_id_value);

  // Check if the session exists
  if (!self->session_manager->hasSession(session_id)) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Session not found"));
    return fl_value_ref(result);
  }

  try {
    // Get the session
    Ort::Session *session = self->session_manager->getSession(session_id);
    if (!session) {
      g_autoptr(FlValue) result = fl_value_new_map();
      fl_value_set_string_take(result, "error", fl_value_new_string("Session is invalid"));
      return fl_value_ref(result);
    }

    // Get all output info
    size_t num_outputs = session->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    g_autoptr(FlValue) result = fl_value_new_list();

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

          // Convert element type to string
          const char *type_str = "unknown";
          switch (element_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            type_str = "float32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            type_str = "uint8";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            type_str = "int8";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            type_str = "uint16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            type_str = "int16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            type_str = "int32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            type_str = "int64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            type_str = "string";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            type_str = "bool";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            type_str = "float16";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            type_str = "double";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            type_str = "uint32";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            type_str = "uint64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            type_str = "complex64";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            type_str = "complex128";
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            type_str = "bfloat16";
            break;
          default:
            type_str = "unknown";
            break;
          }

          fl_value_set_string_take(info_map, "type", fl_value_new_string(type_str));
        } else {
          // Non-tensor type
          fl_value_set_string_take(info_map, "type", fl_value_new_string("non-tensor"));

          // Empty shape for non-tensor types
          FlValue *shape_list = fl_value_new_list();
          fl_value_set_string_take(info_map, "shape", shape_list);
        }

        fl_value_append_take(result, info_map);
      } catch (const Ort::Exception &e) {
        // Skip this output if there's an error
        continue;
      }
    }

    return fl_value_ref(result);
  } catch (const Ort::Exception &e) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string(e.what()));
    return fl_value_ref(result);
  } catch (const std::exception &e) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string(e.what()));
    return fl_value_ref(result);
  }
}

static FlValue *create_ort_value(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Extract arguments
  FlValue *source_type_value = fl_value_lookup_string(args, "sourceType");
  FlValue *data_value = fl_value_lookup_string(args, "data");
  FlValue *shape_value = fl_value_lookup_string(args, "shape");

  // Check if all required arguments are provided
  if (source_type_value == nullptr || data_value == nullptr || shape_value == nullptr ||
      fl_value_get_type(source_type_value) != FL_VALUE_TYPE_STRING ||
      fl_value_get_type(shape_value) != FL_VALUE_TYPE_LIST) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Invalid arguments"));
    return fl_value_ref(result);
  }

  const char *source_type = fl_value_get_string(source_type_value);

  // Convert shape values to vector of int64_t
  size_t shape_size = fl_value_get_length(shape_value);
  std::vector<int64_t> shape;
  for (size_t i = 0; i < shape_size; i++) {
    FlValue *dim = fl_value_get_list_value(shape_value, i);
    if (fl_value_get_type(dim) != FL_VALUE_TYPE_INT) {
      g_autoptr(FlValue) result = fl_value_new_map();
      fl_value_set_string_take(result, "error", fl_value_new_string("Shape must contain integers"));
      return fl_value_ref(result);
    }
    shape.push_back(fl_value_get_int(dim));
  }

  std::string valueId;

  // Handle data according to source type
  if (strcmp(source_type, "float32") == 0) {
    std::vector<float> data_vec;

    // Convert data to vector of floats
    if (fl_value_get_type(data_value) == FL_VALUE_TYPE_FLOAT32_LIST) {
      size_t data_size = fl_value_get_length(data_value);
      // Get direct access to the Dart Float32List data
      const float *float_array = fl_value_get_float32_list(data_value);
      for (size_t i = 0; i < data_size; i++) {
        float val = float_array[i];
        data_vec.push_back(val);
      }
    } else if (fl_value_get_type(data_value) == FL_VALUE_TYPE_LIST) { // regular float list array
      size_t length = fl_value_get_length(data_value);
      data_vec.reserve(length);

      for (size_t i = 0; i < length; i++) {
        FlValue *val = fl_value_get_list_value(data_value, i);
        float float_val = fl_value_get_float(val);
        data_vec.push_back(float_val);
      }
    } else {
      g_autoptr(FlValue) result = fl_value_new_map();
      fl_value_set_string_take(result, "error", fl_value_new_string("Data must be a typed list or a list"));
      return fl_value_ref(result);
    }
    // Create tensor
    valueId = self->tensor_manager->createFloat32Tensor(data_vec, shape);
  } else if (strcmp(source_type, "int32") == 0) {
    std::vector<int32_t> data_vec;

    // Convert data to vector of int32
    if (fl_value_get_type(data_value) == FL_VALUE_TYPE_INT32_LIST) {
      size_t data_size = fl_value_get_length(data_value);
      // Get direct access to the Dart Int32List data
      const int32_t *int_array = fl_value_get_int32_list(data_value);
      for (size_t i = 0; i < data_size; i++) {
        int32_t val = int_array[i];
        data_vec.push_back(val);
      }
    } else if (fl_value_get_type(data_value) == FL_VALUE_TYPE_LIST) { // regular int list array
      size_t length = fl_value_get_length(data_value);
      data_vec.reserve(length);

      for (size_t i = 0; i < length; i++) {
        FlValue *val = fl_value_get_list_value(data_value, i);
        int32_t int_val = static_cast<int32_t>(fl_value_get_int(val));
        data_vec.push_back(int_val);
      }
    } else {
      g_autoptr(FlValue) result = fl_value_new_map();
      fl_value_set_string_take(result, "error", fl_value_new_string("Data must be a typed list or a list"));
      return fl_value_ref(result);
    }
    // Create tensor
    valueId = self->tensor_manager->createInt32Tensor(data_vec, shape);
  } else if (strcmp(source_type, "int64") == 0) {
    std::vector<int64_t> data_vec;

    // Convert data to vector of int64
    if (fl_value_get_type(data_value) == FL_VALUE_TYPE_INT64_LIST) {
      size_t data_size = fl_value_get_length(data_value);
      // Get direct access to the Dart Int64List data
      const int64_t *int_array = fl_value_get_int64_list(data_value);
      for (size_t i = 0; i < data_size; i++) {
        int64_t val = int_array[i];
        data_vec.push_back(val);
      }
    } else if (fl_value_get_type(data_value) == FL_VALUE_TYPE_LIST) { // regular int list array
      size_t length = fl_value_get_length(data_value);
      data_vec.reserve(length);

      for (size_t i = 0; i < length; i++) {
        FlValue *val = fl_value_get_list_value(data_value, i);
        int64_t int_val = fl_value_get_int(val);
        data_vec.push_back(int_val);
      }
    } else {
      g_autoptr(FlValue) result = fl_value_new_map();
      fl_value_set_string_take(result, "error", fl_value_new_string("Data must be a typed list or a list"));
      return fl_value_ref(result);
    }
    // Create tensor
    valueId = self->tensor_manager->createInt64Tensor(data_vec, shape);
  } else if (strcmp(source_type, "uint8") == 0) {
    std::vector<uint8_t> data_vec;

    // Convert data to vector of uint8
    if (fl_value_get_type(data_value) == FL_VALUE_TYPE_UINT8_LIST) {
      size_t data_size = fl_value_get_length(data_value);
      // Get direct access to the Dart Uint8List data
      const uint8_t *uint_array = fl_value_get_uint8_list(data_value);
      for (size_t i = 0; i < data_size; i++) {
        uint8_t val = uint_array[i];
        data_vec.push_back(val);
      }
    } else if (fl_value_get_type(data_value) == FL_VALUE_TYPE_LIST) { // regular int list array
      size_t length = fl_value_get_length(data_value);
      data_vec.reserve(length);

      for (size_t i = 0; i < length; i++) {
        FlValue *val = fl_value_get_list_value(data_value, i);
        uint8_t uint_val = static_cast<uint8_t>(fl_value_get_int(val));
        data_vec.push_back(uint_val);
      }
    } else {
      g_autoptr(FlValue) result = fl_value_new_map();
      fl_value_set_string_take(result, "error", fl_value_new_string("Data must be a typed list or a list"));
      return fl_value_ref(result);
    }
    // Create tensor
    valueId = self->tensor_manager->createUint8Tensor(data_vec, shape);
  } else if (strcmp(source_type, "bool") == 0) {
    std::vector<bool> data_vec;

    // Convert data to vector of bool
    if (fl_value_get_type(data_value) == FL_VALUE_TYPE_LIST) {
      size_t length = fl_value_get_length(data_value);
      data_vec.reserve(length);

      for (size_t i = 0; i < length; i++) {
        FlValue *val = fl_value_get_list_value(data_value, i);
        if (fl_value_get_type(val) == FL_VALUE_TYPE_BOOL) {
          bool bool_val = fl_value_get_bool(val);
          data_vec.push_back(bool_val);
        } else if (fl_value_get_type(val) == FL_VALUE_TYPE_INT) {
          // Handle case where booleans might be represented as integers
          bool bool_val = fl_value_get_int(val) != 0;
          data_vec.push_back(bool_val);
        } else {
          g_autoptr(FlValue) result = fl_value_new_map();
          fl_value_set_string_take(result, "error",
                                   fl_value_new_string("Boolean data must be a list of booleans or integers"));
          return fl_value_ref(result);
        }
      }
    } else {
      g_autoptr(FlValue) result = fl_value_new_map();
      fl_value_set_string_take(result, "error", fl_value_new_string("Boolean data must be a list"));
      return fl_value_ref(result);
    }
    // Create tensor
    valueId = self->tensor_manager->createBoolTensor(data_vec, shape);
  } else {
    // Unsupported source type
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Unsupported source type"));
    return fl_value_ref(result);
  }

  // Check if tensor creation was successful
  if (valueId.empty()) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Failed to create tensor"));
    return fl_value_ref(result);
  }

  // Create response
  g_autoptr(FlValue) result = fl_value_new_map();
  fl_value_set_string_take(result, "valueId", fl_value_new_string(valueId.c_str()));
  fl_value_set_string_take(result, "dataType", fl_value_new_string(source_type));

  // Add shape to response
  FlValue *shape_list = fl_value_new_list();
  for (const auto &dim : shape) {
    fl_value_append_take(shape_list, fl_value_new_int(dim));
  }
  fl_value_set_string_take(result, "shape", shape_list);

  return fl_value_ref(result);
}

static FlValue *convert_ort_value(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Return a new value ID for the converted tensor
  std::lock_guard<std::mutex> lock(self->mutex);

  std::string value_id = self->tensor_manager->generateTensorId();

  g_autoptr(FlValue) result = fl_value_new_map();
  fl_value_set_string_take(result, "valueId", fl_value_new_string(value_id.c_str()));
  fl_value_set_string_take(result, "type", fl_value_new_string("float32"));

  // Add to values map (with null pointer for now)
  self->values[value_id] = nullptr;

  return fl_value_ref(result);
}

static FlValue *move_ort_value_to_device(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Return a new value ID for the moved tensor
  std::lock_guard<std::mutex> lock(self->mutex);

  std::string value_id = self->tensor_manager->generateTensorId();

  g_autoptr(FlValue) result = fl_value_new_map();
  fl_value_set_string_take(result, "valueId", fl_value_new_string(value_id.c_str()));
  fl_value_set_string_take(result, "device", fl_value_new_string("CPU"));

  // Add to values map (with null pointer for now)
  self->values[value_id] = nullptr;

  return fl_value_ref(result);
}

static FlValue *get_ort_value_data(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Extract value ID
  FlValue *value_id_value = fl_value_lookup_string(args, "valueId");

  if (value_id_value == nullptr || fl_value_get_type(value_id_value) != FL_VALUE_TYPE_STRING) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Invalid valueId"));
    return fl_value_ref(result);
  }

  const char *value_id = fl_value_get_string(value_id_value);

  // Get tensor data
  FlValue *tensor_data = self->tensor_manager->getTensorData(value_id);

  // If tensor_data is null, return error
  if (tensor_data == nullptr || fl_value_get_type(tensor_data) == FL_VALUE_TYPE_NULL) {
    g_autoptr(FlValue) result = fl_value_new_map();
    fl_value_set_string_take(result, "error", fl_value_new_string("Tensor not found"));
    if (tensor_data != nullptr) {
      fl_value_unref(tensor_data);
    }
    return fl_value_ref(result);
  }

  return tensor_data; // Already referenced in getTensorData
}

static FlValue *release_ort_value(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Get value ID
  FlValue *value_id_value = fl_value_lookup_string(args, "valueId");

  if (value_id_value == nullptr || fl_value_get_type(value_id_value) != FL_VALUE_TYPE_STRING) {
    return fl_value_new_null();
  }

  const char *value_id = fl_value_get_string(value_id_value);

  // Release tensor
  self->tensor_manager->releaseTensor(value_id);

  return fl_value_new_null();
}