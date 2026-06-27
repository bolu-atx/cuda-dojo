#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
//
// CUDA's runtime API reports errors via return codes, and kernel launches are
// asynchronous so launch errors surface later. Wrap *every* runtime call in
// CUDA_CHECK, and call CUDA_CHECK_KERNEL() right after a <<<>>> launch to catch
// both launch-configuration errors (cudaGetLastError) and execution errors
// (surfaced by the following synchronize).
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t err_ = (expr);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n  in: %s\n  %s\n",          \
                   cudaGetErrorName(err_), __FILE__, __LINE__, #expr,           \
                   cudaGetErrorString(err_));                                   \
      std::abort();                                                             \
    }                                                                           \
  } while (0)

// Synchronizing check — convenient while learning, but note it serializes the
// host with the device, so don't leave it in hot paths once you reach Level 8.
#define CUDA_CHECK_KERNEL()                                                     \
  do {                                                                          \
    CUDA_CHECK(cudaGetLastError());                                             \
    CUDA_CHECK(cudaDeviceSynchronize());                                        \
  } while (0)

namespace dojo {

// ceil_div(a, b): number of blocks of size b needed to cover a items.
constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// RAII timer around CUDA events — measures device-side elapsed time in ms.
class GpuTimer {
public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer &) = delete;
  GpuTimer &operator=(const GpuTimer &) = delete;

  void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

  // Records the stop event, waits for it, and returns elapsed milliseconds.
  float stop(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

// Effective bandwidth in GB/s for moving `bytes` in `ms` milliseconds.
inline double gbps(std::size_t bytes, float ms) {
  return (static_cast<double>(bytes) / 1.0e9) / (ms / 1.0e3);
}

// GFLOP/s for `flops` floating-point ops in `ms` milliseconds.
inline double gflops(std::size_t flops, float ms) {
  return (static_cast<double>(flops) / 1.0e9) / (ms / 1.0e3);
}

inline void print_device_info(int device = 0) {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, device));
  std::printf("GPU %d: %s (sm_%d%d)\n", device, p.name, p.major, p.minor);
  std::printf("  SMs:               %d\n", p.multiProcessorCount);
  std::printf("  Warp size:         %d\n", p.warpSize);
  std::printf("  Max threads/block: %d\n", p.maxThreadsPerBlock);
  std::printf("  Shared mem/block:  %zu KB\n", p.sharedMemPerBlock / 1024);
  std::printf("  Global memory:     %.1f GB\n",
              static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  std::printf("  Peak mem clock:    %.1f GHz, bus %d-bit\n",
              p.memoryClockRate / 1.0e6, p.memoryBusWidth);
}

} // namespace dojo
