
namespace cutl {
template <uint Unroll, typename F, typename Out, typename... Ins>
__global__ void map_unroll(F f, const int n, Out* out, const Ins*... ins) {
  static_assert(Unroll > 0);

  if constexpr (Unroll == 1) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= n) return;

    out[idx] = f(ins[idx]...);
  } else {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int thread_n = blockDim.x;

    if (idx >= n) return;

#pragma unroll
    for (int i = 0; i < Unroll; i++) {
      const int x = idx + i * thread_n;
      if (x < n) out[x] = f(ins[x]...);
    }
  }
}

template <typename F, typename Out, typename... Ins>
__global__ void map(F f, const int n, Out* out, const Ins*... ins) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= n) return;

  out[idx] = f(ins[idx]...);
}

}  // namespace cutl
