#ifndef FLOAT16_UTILS_H
#define FLOAT16_UTILS_H

#include <cstdint>

/**
 * Utility class for float16 conversions
 *
 * Implementation based on the MLAS approach mentioned in OnnxRuntime
 * Reference: https://github.com/microsoft/onnxruntime/commit/a8e776b78bfa0d0b1fec8b34b4545d91c2a9d175
 */
class Float16Utils {
public:
  // Constants for float16 <-> float32 conversion
  static constexpr int FLOAT16_EXPONENT_BIAS = 15;
  static constexpr int FLOAT32_EXPONENT_BIAS = 127;
  static constexpr uint16_t FLOAT16_SIGN_MASK = 0x8000;
  static constexpr uint16_t FLOAT16_EXPONENT_MASK = 0x7C00;
  static constexpr uint16_t FLOAT16_MANTISSA_MASK = 0x03FF;

  /**
   * Convert float32 to float16
   */
  static uint16_t floatToFloat16(float value);

  /**
   * Convert float16 to float32
   */
  static float float16ToFloat(uint16_t value);
};

#endif // FLOAT16_UTILS_H