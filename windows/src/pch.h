#ifndef FLUTTER_ONNXRUNTIME_PCH_H_
#define FLUTTER_ONNXRUNTIME_PCH_H_

// Add commonly included headers to speed up compilation times
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <functional>

// ONNX Runtime headers
#include <onnxruntime_cxx_api.h>

// Windows-specific utilities
#include <shlwapi.h>
#include <shlobj.h>
#include <comdef.h>

// Flutter plugin headers
#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>
#include <flutter/encodable_value.h>

#endif  // FLUTTER_ONNXRUNTIME_PCH_H_ 