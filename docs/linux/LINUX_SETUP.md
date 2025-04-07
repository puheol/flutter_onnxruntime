# Linux Development Setup Guide

This guide provides detailed instructions for setting up your development environment to work on the Flutter ONNX Runtime plugin on Linux.

## Prerequisites

- Ubuntu 20.04 or newer (other distributions should work similarly)
- Flutter SDK (latest stable version recommended)
- CMake 3.10 or newer
- GCC/G++ 9 or newer
- pkg-config

## Installation of Dependencies

### System Dependencies

Install the required system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  libgtk-3-dev \
  liblzma-dev \
  libstdc++-9-dev \
  ninja-build
```

### ONNX Runtime Dependencies

The plugin needs access to ONNX Runtime headers and libraries. The CMakeLists.txt is configured to download and use ONNX Runtime v1.21.0.

If you want to use your system's ONNX Runtime:

```bash
sudo apt-get install -y libonnxruntime-dev
```

You'll need to modify `linux/CMakeLists.txt` to use the system installation instead of downloading it.

## Building and Debugging

### Building the Plugin

1. Navigate to the plugin's root directory:
   ```bash
   cd flutter_onnxruntime
   ```

2. Clean any previous builds:
   ```bash
   flutter clean
   ```

3. Run the Linux example:
   ```bash
   cd example
   flutter run -d linux
   ```

### Debugging Common Issues

#### Missing Headers

If you encounter errors about missing headers:

1. **Flutter Linux headers**: Ensure you have the Flutter Linux development dependencies:
   ```bash
   sudo apt-get install -y libflutter-engine-dev
   ```

2. **glib/gtk headers**: These are required for the Linux plugin:
   ```bash
   sudo apt-get install -y libgtk-3-dev
   ```

3. **ONNX Runtime headers**: Verify the CMakeLists.txt is correctly downloading and including ONNX Runtime headers.

#### Linking Errors

If you encounter linking errors:

1. Check that the ONNX Runtime libraries are correctly located.
2. Verify that the CMakeLists.txt is correctly linking against them.
3. Check for compatibility issues between the ONNX Runtime version and your code.

## Development Workflow

1. **Make your changes**: Edit the files in `linux/` directory for the Linux implementation.

2. **Test your changes**:
   ```bash
   cd example
   flutter clean
   flutter run -d linux
   ```

3. **Debug with print statements**:
   - Add `g_print("Debug: %s\n", "your message");` statements to debug C++ code.
   - Use `print('Debug: your message');` in Dart code.

4. **Check logs**:
   - Run with `--verbose` flag to see more detailed logs:
     ```bash
     flutter run -d linux --verbose
     ```

5. **Memory management**:
   - Be careful with `FlValue` reference counting.
   - Always unref `FlValue` objects when you're done with them.
   - For ONNX Runtime objects, follow their ownership patterns (usually RAII with C++ smart pointers).

## FlValue Reference Counting

A common issue is improper FlValue reference counting. Remember:

- `fl_value_new_*()` creates a new FlValue with a reference count of 1
- `fl_value_ref()` increases the reference count
- `fl_value_unref()` decreases the reference count
- When the reference count reaches 0, the FlValue is freed

Always balance your refs and unrefs to avoid memory leaks and crashes.