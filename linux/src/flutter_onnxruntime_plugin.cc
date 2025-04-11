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
#include "value_conversion.h"

#define FLUTTER_ONNXRUNTIME_PLUGIN(obj)                                                                                \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), flutter_onnxruntime_plugin_get_type(), FlutterOnnxruntimePlugin))

struct _FlutterOnnxruntimePlugin {
  GObject parent_instance;

  // SessionManager for handling ONNX Runtime sessions
  SessionManager *session_manager;

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
  self->next_value_id = 1;
}

static void flutter_onnxruntime_plugin_dispose(GObject *object) {
  FlutterOnnxruntimePlugin *self = FLUTTER_ONNXRUNTIME_PLUGIN(object);

  // Clean up session manager and values
  delete self->session_manager;

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

  // Create dummy metadata response
  g_autoptr(FlValue) result = fl_value_new_map();
  fl_value_set_string_take(result, "producerName", fl_value_new_string("ONNX Runtime Dummy Producer"));
  fl_value_set_string_take(result, "graphName", fl_value_new_string("Dummy Graph"));
  fl_value_set_string_take(result, "domain", fl_value_new_string("ai.onnx"));
  fl_value_set_string_take(result, "description", fl_value_new_string("Dummy model description"));
  fl_value_set_string_take(result, "version", fl_value_new_int(1));

  // Create custom metadata map
  g_autoptr(FlValue) custom_metadata_map = fl_value_new_map();
  fl_value_set_string_take(custom_metadata_map, "dummy_key", fl_value_new_string("dummy_value"));
  fl_value_set_string(result, "customMetadataMap", custom_metadata_map);

  return fl_value_ref(result);
}

static FlValue *get_input_info(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Return dummy input info
  g_autoptr(FlValue) input = fl_value_new_map();
  fl_value_set_string_take(input, "name", fl_value_new_string("input"));
  fl_value_set_string_take(input, "type", fl_value_new_string("float"));

  g_autoptr(FlValue) shape = fl_value_new_list();
  fl_value_append_take(shape, fl_value_new_int(1));
  fl_value_append_take(shape, fl_value_new_int(3));
  fl_value_append_take(shape, fl_value_new_int(224));
  fl_value_append_take(shape, fl_value_new_int(224));
  fl_value_set_string(input, "shape", shape);

  g_autoptr(FlValue) result = fl_value_new_list();
  fl_value_append(result, input);

  return fl_value_ref(result);
}

static FlValue *get_output_info(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Return dummy output info
  g_autoptr(FlValue) output = fl_value_new_map();
  fl_value_set_string_take(output, "name", fl_value_new_string("output"));
  fl_value_set_string_take(output, "type", fl_value_new_string("float"));

  g_autoptr(FlValue) shape = fl_value_new_list();
  fl_value_append_take(shape, fl_value_new_int(1));
  fl_value_append_take(shape, fl_value_new_int(1000));
  fl_value_set_string(output, "shape", shape);

  g_autoptr(FlValue) result = fl_value_new_list();
  fl_value_append(result, output);

  return fl_value_ref(result);
}

static FlValue *create_ort_value(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Dummy implementation that returns a new value ID
  std::lock_guard<std::mutex> lock(self->mutex);

  std::string value_id = "value_" + std::to_string(self->next_value_id++);

  g_autoptr(FlValue) result = fl_value_new_map();
  fl_value_set_string_take(result, "valueId", fl_value_new_string(value_id.c_str()));
  fl_value_set_string_take(result, "type", fl_value_new_string("float"));

  // Add to values map (with null pointer for now)
  self->values[value_id] = nullptr;

  return fl_value_ref(result);
}

static FlValue *convert_ort_value(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Return a new value ID for the converted tensor
  std::lock_guard<std::mutex> lock(self->mutex);

  std::string value_id = "value_" + std::to_string(self->next_value_id++);

  g_autoptr(FlValue) result = fl_value_new_map();
  fl_value_set_string_take(result, "valueId", fl_value_new_string(value_id.c_str()));
  fl_value_set_string_take(result, "type", fl_value_new_string("float"));

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
  // Return dummy tensor data
  g_autoptr(FlValue) data = fl_value_new_list();
  for (int i = 0; i < 10; i++) {
    fl_value_append_take(data, fl_value_new_float(i * 0.1));
  }

  g_autoptr(FlValue) shape = fl_value_new_list();
  fl_value_append_take(shape, fl_value_new_int(1));
  fl_value_append_take(shape, fl_value_new_int(10));

  g_autoptr(FlValue) result = fl_value_new_map();
  fl_value_set_string(result, "data", data);
  fl_value_set_string(result, "shape", shape);
  fl_value_set_string_take(result, "type", fl_value_new_string("float"));

  return fl_value_ref(result);
}

static FlValue *release_ort_value(FlutterOnnxruntimePlugin *self, FlValue *args) {
  // Get value ID
  FlValue *value_id_value = fl_value_lookup_string(args, "valueId");
  const char *value_id = fl_value_get_string(value_id_value);

  std::lock_guard<std::mutex> lock(self->mutex);

  // Remove from values map
  self->values.erase(value_id);

  return fl_value_new_null();
}