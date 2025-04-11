#ifndef VALUE_CONVERSION_H
#define VALUE_CONVERSION_H

#include <flutter_linux/flutter_linux.h>
#include <map>
#include <string>
#include <vector>

// Convert a C++ vector to a FlValue
template <typename T> FlValue *vector_to_fl_value(const std::vector<T> &vec);

// Specialization for string vectors
template <> FlValue *vector_to_fl_value<std::string>(const std::vector<std::string> &vec);

// Specialization for int vectors
template <> FlValue *vector_to_fl_value<int>(const std::vector<int> &vec);

// Specialization for float vectors
template <> FlValue *vector_to_fl_value<float>(const std::vector<float> &vec);

// Convert a FlValue map to a C++ map
std::map<std::string, FlValue *> fl_value_to_map(FlValue *map_value);

#endif // VALUE_CONVERSION_H