#include "./api.hpp"

namespace cpu {

static int collatz(int in) {
  for (int i = 0; i < 1024; i++) {
    in = (in % 2 == 0) ? in / 2 : 3 * in + 1;
  }
  return in;
}

CollatzMap_1024::CollatzMap_1024() {}
CollatzMap_1024::~CollatzMap_1024() {}
void CollatzMap_1024::call(const int* in, int* out, int n) {
  for (int i = 0; i < n; i++) {
    out[i] = collatz(in[i]);
  }
}
}  // namespace cpu