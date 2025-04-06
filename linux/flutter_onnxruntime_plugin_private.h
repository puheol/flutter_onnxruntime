#include <flutter_linux/flutter_linux.h>
#include <map>
#include <string>
#include <vector>

#include "include/flutter_onnxruntime/flutter_onnxruntime_plugin.h"

// This file exposes some plugin internals for unit testing. See
// https://github.com/flutter/flutter/issues/88724 for current limitations
// in the unit-testable API.

// Handles the getPlatformVersion method call.
FlMethodResponse *get_platform_version();

// Helper function to generate a unique session ID
std::string generate_uuid();

// Helper function to convert FlValue to std::vector - implemented with
// specializations
template <typename T> std::vector<T> fl_value_to_vector(FlValue *value);

// Helper function to convert std::vector to FlValue list - implemented with
// specializations
template <typename T> FlValue *vector_to_fl_value(const std::vector<T> &vec);

// Template specializations for vector_to_fl_value
template <> FlValue *vector_to_fl_value<std::string>(const std::vector<std::string> &vec);

template <> FlValue *vector_to_fl_value<int64_t>(const std::vector<int64_t> &vec);

template <> FlValue *vector_to_fl_value<float>(const std::vector<float> &vec);

template <> FlValue *vector_to_fl_value<int>(const std::vector<int> &vec);

// Helper function to convert FlValue map to std::map
std::map<std::string, FlValue *> fl_value_to_map(FlValue *value);
