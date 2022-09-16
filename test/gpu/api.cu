#include <cuda_runtime_api.h>

#include <algorithm>

#include "../util.hpp"
#include "./api.hpp"
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

CollatzMap_1024::CollatzMap_1024() {
  for (int i = 0; i < StreamN; i++) cudaMalloc(&block[i], sizeof(int) * BlockN);
}
CollatzMap_1024::~CollatzMap_1024() {
  for (int i = 0; i < StreamN; i++) cudaFree(block[i]);
}
void CollatzMap_1024::call(const int* in, int* out, int n) {
  cudaStream_t ss[StreamN];
  for (int i = 0; i < StreamN; i++) cudaStreamCreate(ss + i);
  for (int i = 0; i < n; i += BlockN) {
    const int sid = i % StreamN;
    cudaMemcpyAsync(block[sid], in + i, sizeof(int) * BlockN,
                    cudaMemcpyHostToDevice, ss[sid]);
    cutl::map_unroll<2><<<1, BlockN / 2, 0, ss[sid]>>>(
        Collatz<BlockN>(), block[sid], block[sid], min(BlockN, n - i));
    cudaMemcpyAsync(out + i, block[sid], sizeof(int) * min(BlockN, n - i),
                    cudaMemcpyDeviceToHost, ss[sid]);
  }
}
}  // namespace gpu