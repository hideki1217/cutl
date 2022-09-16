
namespace cutl {
template <uint Unroll, typename F, typename In, typename Out>
__global__ void map_unroll(F f, const In* in, Out* out, const int n) {
  static_assert(Unroll > 0);
  static_assert(Unroll > 1, "call map instead");
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int thread_n = blockDim.x;

  if (idx >= n) return;

  const int m = min(Unroll, n - idx);
#pragma unroll
  for (int i = 0; i < m; i++) {
    const int x = idx + i * thread_n;
    out[x] = f(in[x]);
  }
}

template <typename F, typename In, typename Out>
__global__ void map(F f, const In* in, Out* out, const int n) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= n) return;

  out[idx] = f(in[idx]);
}
}  // namespace cutl
