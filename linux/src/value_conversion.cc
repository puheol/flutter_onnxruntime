#include "value_conversion.h"

// Implementation of the vector_to_fl_value specialization for strings
template <> FlValue *vector_to_fl_value<std::string>(const std::vector<std::string> &vec) {
  FlValue *list = fl_value_new_list();

  for (const auto &str : vec) {
    fl_value_append_take(list, fl_value_new_string(str.c_str()));
  }

  return list;
}

// Implementation of the vector_to_fl_value specialization for ints
template <> FlValue *vector_to_fl_value<int>(const std::vector<int> &vec) {
  FlValue *list = fl_value_new_list();

  for (const auto &val : vec) {
    fl_value_append_take(list, fl_value_new_int(val));
  }

  return list;
}

// Implementation of the vector_to_fl_value specialization for floats
template <> FlValue *vector_to_fl_value<float>(const std::vector<float> &vec) {
  FlValue *list = fl_value_new_list();

  for (const auto &val : vec) {
    fl_value_append_take(list, fl_value_new_float(val));
  }

  return list;
}

// Implementation of fl_value_to_map
std::map<std::string, FlValue *> fl_value_to_map(FlValue *map_value) {
  std::map<std::string, FlValue *> result;

  if (map_value == nullptr || fl_value_get_type(map_value) != FL_VALUE_TYPE_MAP) {
    return result;
  }

  size_t size = fl_value_get_length(map_value);

  for (size_t i = 0; i < size; i++) {
    FlValue *key = fl_value_get_map_key(map_value, i);
    FlValue *value = fl_value_get_map_value(map_value, i);

    if (fl_value_get_type(key) == FL_VALUE_TYPE_STRING) {
      result[fl_value_get_string(key)] = value;
    }
  }

  return result;
}