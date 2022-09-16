#ifdef GPU
#define PROC_NAME gpu
#else
#define PROC_NAME cpu
#endif

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "cpu/api.hpp"
#include "gpu/api.hpp"

template <typename T>
bool is_same(const T* a, const T* b, const int n) {
  for (int i = 0; i < n; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
template <>
bool is_same(const float* a, const float* b, const int n) {
  for (int i = 0; i < n; i++) {
    if (std::abs(a[i] - b[i]) >= 1e-5) return false;
  }
  return true;
}

class Timer {
 public:
  void start() {
    m_StartTime = std::chrono::system_clock::now();
    m_bRunning = true;
  }

  void stop() {
    m_EndTime = std::chrono::system_clock::now();
    m_bRunning = false;
  }

  double elapsed_ms() {
    std::chrono::time_point<std::chrono::system_clock> endTime;

    if (m_bRunning) {
      endTime = std::chrono::system_clock::now();
    } else {
      endTime = m_EndTime;
    }

    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                                 m_StartTime)
        .count();
  }

  double elapsed_s() { return elapsed_ms() / 1000.0; }

 private:
  std::chrono::time_point<std::chrono::system_clock> m_StartTime;
  std::chrono::time_point<std::chrono::system_clock> m_EndTime;
  bool m_bRunning = false;
};

int main() {
  const int n = 2000000;
  auto in = std::make_unique<int[]>(n);
  for (int i = 0; i < n; i++) in[i] = i + 1;

  Timer timer;

  timer.start();
  auto out_cpu = std::make_unique<int[]>(n);
  {
    auto collatz = cpu::CollatzMap_1024();

    collatz.call(in.get(), out_cpu.get(), n);
  }
  std::cout << "end cpu" << std::endl;
  timer.stop();
  std::cout << "cpu: " << timer.elapsed_ms() << "(ms)" << std::endl;

  timer.start();
  auto out_gpu = std::make_unique<int[]>(n);
  {
    auto collatz = gpu::CollatzMap_1024();

    collatz.call(in.get(), out_gpu.get(), n);
  }
  std::cout << "end gpu" << std::endl;
  timer.stop();
  std::cout << "gpu: " << timer.elapsed_ms() << "(ms)" << std::endl;

  assert(is_same(out_cpu.get(), out_gpu.get(), n));

  return 0;
}