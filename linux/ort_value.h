#ifndef ORT_VALUE_H
#define ORT_VALUE_H

#include <onnxruntime_cxx_api.h>
#include <string>

// Helper function to generate a UUID
std::string generate_uuid();

// C API for OrtValue functions - these can be called from the Flutter FFI
extern "C" {
// Create a new OrtValue from data
// Returns a JSON string with tensor info or nullptr on error
char *ort_create_tensor(const char *source_type, // Source data type (e.g., "float32", "int32")
                        const void *data,        // Pointer to the data
                        const int64_t *shape,    // Array of shape dimensions
                        int shape_len,           // Number of dimensions
                        const char *target_type, // Target data type (optional, can be null)
                        const char *device,      // Target device (optional, can be null)
                        char **error_out         // Error message (out parameter)
);

// Convert an OrtValue to a different data type
// Returns a JSON string with tensor info or nullptr on error
char *ort_convert_tensor(const char *value_id,    // ID of the OrtValue to convert
                         const char *target_type, // Target data type
                         char **error_out         // Error message (out parameter)
);

// Move an OrtValue to a different device
// Returns a JSON string with tensor info or nullptr on error
char *ort_move_tensor_to_device(const char *value_id,      // ID of the OrtValue to move
                                const char *target_device, // Target device
                                char **error_out           // Error message (out parameter)
);

// Get data from an OrtValue
// Returns a JSON string with data and shape or nullptr on error
char *ort_get_tensor_data(const char *value_id,  // ID of the OrtValue to get data from
                          const char *data_type, // Requested data type
                          char **error_out       // Error message (out parameter)
);

// Release an OrtValue
// Returns true on success, false on error
bool ort_release_tensor(const char *value_id, // ID of the OrtValue to release
                        char **error_out      // Error message (out parameter)
);
}

#endif // ORT_VALUE_H