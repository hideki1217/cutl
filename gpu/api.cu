#include "./api.hpp"

#include <algorithm>
#include <cuda_runtime_api.h>

#include "util.hpp"
#include "cutl.cuh"

namespace gpu {

template <int N>
class Collatz {
 public:
  __device__ __host__ inline int operator()(int x) const {
    for (int i = 0; i < N; i++) {
      x = (x % 2 == 0) ? x / 2 : 3 * x + 1;
    }
    return x;
  }
};

static constexpr int BlockN = 128;
CollatzMap::CollatzMap() {
  for (int i = 0; i < StreamN; i++) cudaMalloc((void**)&tile[i], sizeof(int) * TileN);
}
CollatzMap::~CollatzMap() {
  for (int i = 0; i < StreamN; i++) cudaFree(tile[i]);
}
void CollatzMap::call(const int* in, int* out, int n) {
  cudaStream_t ss[StreamN];
  for (int i = 0; i < StreamN; i++) cudaStreamCreate(ss + i);
  for (int i = 0; i < n; i += TileN) {
    const int sid = i % StreamN;
    cudaMemcpyAsync(tile[sid], in + i, sizeof(int) * TileN,
                    cudaMemcpyHostToDevice, ss[sid]);
    cutl::map<<<div_up(TileN, BlockN), BlockN, 0, ss[sid]>>>(
        Collatz<1024>(), tile[sid], tile[sid], min(TileN, n - i));
    cutl::map<<<div_up(TileN, BlockN), BlockN, 0, ss[sid]>>>(
        Collatz<1024>(), tile[sid], tile[sid], min(TileN, n - i));
    cutl::map<<<div_up(TileN, BlockN), BlockN, 0, ss[sid]>>>(
        Collatz<1024>(), tile[sid], tile[sid], min(TileN, n - i));
    cudaMemcpyAsync(out + i, tile[sid], sizeof(int) * util::min(TileN, n - i),
                    cudaMemcpyDeviceToHost, ss[sid]);
  }
  for (int i = 0; i < StreamN; i++) cudaStreamDestroy(ss[i]);
}
}  // namespace gpu