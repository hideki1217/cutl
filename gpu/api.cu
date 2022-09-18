#include <cuda_runtime_api.h>

#include <algorithm>

#include "./api.hpp"
#include "cutl.cuh"
#include "util.hpp"

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
  for (int i = 0; i < StreamN; i++)
    cudaMalloc((void**)&tile[i], sizeof(int) * TileN);
  for (int i = 0; i < StreamN; i++)
    cudaStreamCreateWithFlags(ss + i, cudaStreamNonBlocking);
}
CollatzMap::~CollatzMap() {
  for (int i = 0; i < StreamN; i++) cudaFree(tile[i]);
  for (int i = 0; i < StreamN; i++) cudaStreamDestroy(ss[i]);
}
void CollatzMap::call(const int* in, int* out, int n) {
  for (int i = 0; i < n; i += TileN) {
    const int sid = i % StreamN;
    cudaMemcpyAsync(tile[sid], in + i, sizeof(int) * TileN,
                    cudaMemcpyHostToDevice, ss[sid]);
    cutl::map_unroll<4><<<div_up(TileN, BlockN), BlockN / 4, 0, ss[sid]>>>(
        Collatz<1024>(), min(TileN, n - i), tile[sid], tile[sid]);
    cudaMemcpyAsync(out + i, tile[sid], sizeof(int) * min(TileN, n - i),
                    cudaMemcpyDeviceToHost, ss[sid]);
  }
}
}  // namespace gpu