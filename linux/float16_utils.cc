#include "float16_utils.h"
#include <cmath>
#include <cstring>

uint16_t Float16Utils::floatToFloat16(float value) {
  // Get float bits as int for bit manipulation
  uint32_t floatBits;
  std::memcpy(&floatBits, &value, sizeof(float));

  // Extract sign, exponent, and mantissa from float32
  uint16_t sign = (floatBits >> 31) & 0x1;
  int exponent = ((floatBits >> 23) & 0xFF) - FLOAT32_EXPONENT_BIAS + FLOAT16_EXPONENT_BIAS;
  uint32_t mantissa = floatBits & 0x7FFFFF;

  // Handle special cases
  if (exponent <= 0) {
    // Zero or denormal
    return (sign << 15);
  } else if (exponent >= 31) {
    // Infinity or NaN
    if (mantissa == 0) {
      // Infinity
      return (sign << 15) | FLOAT16_EXPONENT_MASK;
    } else {
      // NaN
      return (sign << 15) | FLOAT16_EXPONENT_MASK | 0x200;
    }
  }

  // Regular numbers
  uint16_t float16Bits = (sign << 15) | (exponent << 10) | (mantissa >> 13);
  return float16Bits;
}

float Float16Utils::float16ToFloat(uint16_t value) {
  uint16_t float16Bits = value;

  // Extract sign, exponent, and mantissa from float16
  uint32_t sign = (float16Bits & FLOAT16_SIGN_MASK) << 16;
  uint32_t exponent = (float16Bits & FLOAT16_EXPONENT_MASK) >> 10;
  uint32_t mantissa = float16Bits & FLOAT16_MANTISSA_MASK;

  // Handle special cases
  if (exponent == 0) {
    // Zero or denormal
    if (mantissa == 0) {
      float result;
      uint32_t float32Bits = sign;
      std::memcpy(&result, &float32Bits, sizeof(float));
      return result;
    }

    // Denormal - convert to normal
    int e = 1;
    uint32_t m = mantissa;
    while ((m & 0x400) == 0) {
      m = m << 1;
      e++;
    }

    uint32_t normalizedExponent = exponent - e + 1;
    uint32_t float32Bits =
        sign | ((normalizedExponent + FLOAT32_EXPONENT_BIAS - FLOAT16_EXPONENT_BIAS) << 23) | ((m & 0x3FF) << 13);

    float result;
    std::memcpy(&result, &float32Bits, sizeof(float));
    return result;
  } else if (exponent == 31) {
    // Infinity or NaN
    uint32_t float32Bits;
    if (mantissa == 0) {
      // Infinity
      float32Bits = sign | 0x7F800000;
    } else {
      // NaN
      float32Bits = sign | 0x7FC00000;
    }

    float result;
    std::memcpy(&result, &float32Bits, sizeof(float));
    return result;
  }

  // Regular numbers
  uint32_t float32Bits = sign | ((exponent + FLOAT32_EXPONENT_BIAS - FLOAT16_EXPONENT_BIAS) << 23) | (mantissa << 13);

  float result;
  std::memcpy(&result, &float32Bits, sizeof(float));
  return result;
}