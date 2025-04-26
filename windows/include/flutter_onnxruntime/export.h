#ifndef FLUTTER_ONNXRUNTIME_EXPORT_H_
#define FLUTTER_ONNXRUNTIME_EXPORT_H_

#ifdef FLUTTER_PLUGIN_IMPL
#define FLUTTER_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FLUTTER_PLUGIN_EXPORT __declspec(dllimport)
#endif

#endif // FLUTTER_ONNXRUNTIME_EXPORT_H_