#include <cuda_runtime_api.h>

namespace gpu {

template <typename T>
using device_ptr = T*;

class CollatzMap {
  static constexpr int TileN = 4096;
  static constexpr int StreamN = 3;

 private:
  device_ptr<int> tile[StreamN];
  cudaStream_t ss[StreamN];

 public:
  CollatzMap();
  ~CollatzMap();
  void call(const int* in, int* out, int n);
};

}  // namespace gpu