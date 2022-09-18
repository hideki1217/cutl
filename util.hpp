#pragma once

#define div_up(a, b) (((a) + (b)-1) / (b))

namespace util {
template <typename T>
__device__ __host__ T min(T a, T b) {
  return (a > b) ? b : a;
}

template <typename T>
__device__ __host__ T max(T a, T b) {
  return (a < b) ? b : a;
}
}  // namespace util
