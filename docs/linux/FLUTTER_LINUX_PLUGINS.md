# Flutter Linux Plugin Architecture Guide

This document explains the architecture of Flutter plugins for Linux and how to work with the Flutter Linux plugin API.

## Plugin Architecture

Flutter Linux plugins follow a C-based architecture that integrates with Flutter's embedder API. The key components are:

1. **Plugin Class**: A GObject-derived class that implements the `FlPlugin` interface.
2. **Method Channel**: Handles method calls from the Dart side of the plugin.
3. **Registration Function**: Registers the plugin with the Flutter engine.

## Basic Plugin Structure

A typical plugin has this structure:

```c
// Plugin definition
struct _FlutterOnnxruntimePlugin {
  GObject parent_instance;
  // Plugin-specific members
};

G_DEFINE_TYPE(FlutterOnnxruntimePlugin, flutter_onnxruntime_plugin, fl_plugin_get_type())

// Method handler
static FlMethodResponse* method_call_cb(FlPlugin* plugin,
                                        FlMethodCall* method_call,
                                        gpointer user_data) {
  // Handle method calls here
}

// Plugin initialization
static void flutter_onnxruntime_plugin_init(FlutterOnnxruntimePlugin* self) {}

// Plugin class initialization
static void flutter_onnxruntime_plugin_class_init(FlutterOnnxruntimePluginClass* klass) {
  FL_PLUGIN_CLASS(klass)->handle_method_call = method_call_cb;
}

// Plugin registration function
void flutter_onnxruntime_plugin_register_with_registrar(FlPluginRegistrar* registrar) {
  FlutterOnnxruntimePlugin* plugin = FLUTTER_ONNXRUNTIME_PLUGIN(
      g_object_new(flutter_onnxruntime_plugin_get_type(), nullptr));
  fl_plugin_register_with_registrar(FL_PLUGIN(plugin), registrar);
}
```

## Working with FlValue

`FlValue` is the C type used to represent Dart values in the Flutter Linux API. It's reference-counted and has several types:

- `FL_VALUE_TYPE_NULL`
- `FL_VALUE_TYPE_BOOL`
- `FL_VALUE_TYPE_INT`
- `FL_VALUE_TYPE_FLOAT`
- `FL_VALUE_TYPE_STRING`
- `FL_VALUE_TYPE_MAP`
- `FL_VALUE_TYPE_LIST`

### Reference Counting

`FlValue` uses reference counting for memory management:

```c
// Create a new FlValue
FlValue* value = fl_value_new_string("hello");  // ref count = 1

// Increase reference count
fl_value_ref(value);  // ref count = 2

// Decrease reference count
fl_value_unref(value);  // ref count = 1

// Use it somewhere
// ...

// Finally release it
fl_value_unref(value);  // ref count = 0, memory is freed
```

### Common Operations

```c
// Create values
FlValue* null_val = fl_value_new_null();
FlValue* bool_val = fl_value_new_bool(TRUE);
FlValue* int_val = fl_value_new_int(42);
FlValue* float_val = fl_value_new_float(3.14);
FlValue* string_val = fl_value_new_string("hello");

// Create a list
FlValue* list_val = fl_value_new_list();
fl_value_append(list_val, fl_value_new_int(1));
fl_value_append(list_val, fl_value_new_int(2));

// Create a map
FlValue* map_val = fl_value_new_map();
fl_value_set(map_val, fl_value_new_string("key"), fl_value_new_string("value"));

// Get values
int64_t int_value = fl_value_get_int(int_val);
const char* string_value = fl_value_get_string(string_val);
```

### Converting Between C++ and FlValue

When working with C++ code and FlValue, you'll need to convert between them. Here are common patterns:

```cpp
// Convert std::string to FlValue
FlValue* string_to_fl_value(const std::string& str) {
  return fl_value_new_string(str.c_str());
}

// Convert FlValue to std::string
std::string fl_value_to_string(FlValue* value) {
  return fl_value_get_string(value);
}

// Convert std::vector<T> to FlValue list
template <typename T>
FlValue* vector_to_fl_value(const std::vector<T>& vec);

// Specialization for std::string
template <>
FlValue* vector_to_fl_value(const std::vector<std::string>& vec) {
  FlValue* list = fl_value_new_list();
  for (const auto& str : vec) {
    fl_value_append(list, fl_value_new_string(str.c_str()));
  }
  return list;
}

// Convert std::map to FlValue map
template <typename K, typename V>
FlValue* map_to_fl_value(const std::map<K, V>& map);
```

## Method Channel Communication

Method channels are used to communicate between Dart and native code:

```c
// Handle method call
static FlMethodResponse* method_call_cb(FlPlugin* plugin,
                                        FlMethodCall* method_call,
                                        gpointer user_data) {
  const gchar* method = fl_method_call_get_name(method_call);
  FlValue* args = fl_method_call_get_args(method_call);
  
  if (strcmp(method, "myMethod") == 0) {
    // Extract arguments
    FlValue* arg = fl_value_lookup_string(args, "parameter");
    
    // Do something with the arguments
    // ...
    
    // Return a success response
    FlValue* result = fl_value_new_string("success");
    return FL_METHOD_RESPONSE(fl_method_success_response_new(result));
  } else {
    // Return a not implemented response
    return FL_METHOD_RESPONSE(fl_method_not_implemented_response_new());
  }
}
```

## Error Handling

To report errors back to Dart:

```c
// Create an error response
FlMethodResponse* CreateErrorResponse(const char* code, const char* message) {
  FlValue* details = fl_value_new_map();
  return FL_METHOD_RESPONSE(fl_method_error_response_new(code, message, details));
}

// Usage
if (error_occurred) {
  return CreateErrorResponse("ERROR_CODE", "Something went wrong");
}
```

## Best Practices

1. **Always handle reference counting properly**:
   - Every `fl_value_new_*` call should have a corresponding `fl_value_unref`.
   - When returning values, the caller takes ownership of the reference.

2. **Use wrapper functions for complex conversions**:
   - Create helper functions to convert between C++ types and FlValue.

3. **Error handling**:
   - Always return appropriate error responses with meaningful error codes.
   - Include detailed error messages to help debugging.

4. **Memory safety**:
   - Check for null pointers before dereferencing.
   - Use G_DEFINE_TYPE to ensure proper cleanup.

## Additional Resources

- [Flutter Linux Embedding Source Code](https://github.com/flutter/engine/tree/master/shell/platform/linux)
- [GObject Documentation](https://docs.gtk.org/gobject/) 