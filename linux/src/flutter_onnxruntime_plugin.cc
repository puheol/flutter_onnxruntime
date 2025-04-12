#include "include/flutter_onnxruntime/flutter_onnxruntime_plugin.h"

#include <flutter_linux/flutter_linux.h>
#include <gtk/gtk.h>
#include <sys/utsname.h>

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>

#include "session_manager.h"
#include "tensor_manager.h"
#include "value_conversion.h"

#define FLUTTER_ONNXRUNTIME_PLUGIN(obj)                                                                                \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), flutter_onnxruntime_plugin_get_type(), FlutterOnnxruntimePlugin))

struct _FlutterOnnxruntimePlugin {
  GObject parent_instance;

  // SessionManager for handling ONNX Runtime sessions
  SessionManager *session_manager;

  // TensorManager for handling OrtValue objects
  TensorManager *tensor_manager;

  // Value ID counter for generating unique value IDs
  int next_value_id;

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
  self->next_value_id = 1;
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

  // Create session using session manager
  std::string session_id = self->session_manager->createSession(model_path, nullptr);

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
  // Dummy implementation that returns a fixed "output" tensor ID
  std::lock_guard<std::mutex> lock(self->mutex);

  std::string value_id = "value_" + std::to_string(self->next_value_id++);

  g_autoptr(FlValue) outputs_map = fl_value_new_map();
  fl_value_set_string_take(outputs_map, "output", fl_value_new_string(value_id.c_str()));

  g_autoptr(FlValue) result = fl_value_new_map();
  fl_value_set_string(result, "outputs", outputs_map);
  fl_value_set_string_take(result, "status", fl_value_new_string("success"));

  // Add to values map (with null pointer for now)
  self->values[value_id] = nullptr;

  return fl_value_ref(result);
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
  std::vector<float> data_vec;

  // Handle data according to source type
  if (strcmp(source_type, "float32") == 0) {
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
  fl_value_set_string_take(result, "dataType",
                           fl_value_new_string(strcmp(source_type, "float32") == 0 ? "float32" : source_type));

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

  std::string value_id = "value_" + std::to_string(self->next_value_id++);

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

  std::string value_id = "value_" + std::to_string(self->next_value_id++);

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