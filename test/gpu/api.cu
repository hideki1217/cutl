#include "./api.hpp"

#include <algorithm>
#include <cuda_runtime_api.h>

#include "../util.hpp"
#include "cutl.cuh"

namespace gpu{

template<int N>
class Collatz {
public:
    __device__ __host__ inline
    int operator() (int x) const {
        for(int i=0; i<N; i++) {
            x = (x%2 == 0) ? x/2 : 3*x+1;
        }
        return x;
    }
};

CollatzMap_1024::CollatzMap_1024()  { cudaMalloc(&block, sizeof(int) * BlockN); }
CollatzMap_1024::~CollatzMap_1024() { cudaFree(block); }
void CollatzMap_1024::call(const int* in, int* out, int n) {
        for(int i=0; i<n; i+=BlockN){
            cudaMemcpy(block, in + i, sizeof(int) * BlockN, cudaMemcpyHostToDevice);
            cutl::map_unroll<2><<<1, BlockN/2>>>(Collatz<BlockN>(), block, block, min(BlockN, n - i));
            cudaMemcpy(out + i, block, sizeof(int) * min(BlockN, n - i), cudaMemcpyDeviceToHost);
        }
    }
}