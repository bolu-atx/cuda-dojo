#include "level05.cuh"

#include "dojo/cuda_utils.cuh"

#include <cstdio>
#include <vector>

// Level 5 demo — the deliverable. For each kernel we compute arithmetic
// intensity BEFORE timing, then measure throughput and compare against the GPU's
// peak bandwidth. The conclusion you should be able to state out loud: these
// stencils are memory-bound, so the only lever is moving bytes more efficiently.
//
// NOTE: achieved GB/s here counts the *minimum* traffic (read the image once,
// write it once). The naive kernels actually move more than that because they
// re-fetch overlapping windows — `ncu --set full` reports the true DRAM bytes.
// The gap between this number and peak is your optimization headroom.

namespace {

// Theoretical peak DRAM bandwidth from device properties:
//   2 (DDR) * memory_clock(Hz) * bus_width_bytes.
double peak_gbps(int device = 0) {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, device));
  double clock_hz = static_cast<double>(p.memoryClockRate) * 1.0e3; // kHz -> Hz
  double bus_bytes = p.memoryBusWidth / 8.0;
  return 2.0 * clock_hz * bus_bytes / 1.0e9;
}

void report(const char *name, float ms, std::size_t img_bytes,
            double flops_per_pixel, double read_bytes_per_pixel, std::size_t n,
            double peak) {
  double moved = 2.0 * static_cast<double>(img_bytes); // min: 1 read + 1 write
  double achieved = dojo::gbps(static_cast<std::size_t>(moved), ms);
  double ai = flops_per_pixel / read_bytes_per_pixel; // FLOP per byte read
  double gflops = dojo::gflops(
      static_cast<std::size_t>(flops_per_pixel * static_cast<double>(n)), ms);
  std::printf("%-16s %.3f ms | AI=%.2f FLOP/B (%s) | %.0f GFLOP/s | "
              "~%.0f GB/s = %.0f%% of peak\n",
              name, ms, ai, ai < 1.0 ? "memory-bound" : "compute-bound", gflops,
              achieved, 100.0 * achieved / peak);
}

} // namespace

int main() {
  dojo::print_device_info();
  const double peak = peak_gbps();
  std::printf("Theoretical peak bandwidth: ~%.0f GB/s\n\n", peak);

  const int w = 4096, h = 4096;
  const std::size_t n = static_cast<std::size_t>(w) * h;
  const std::size_t bytes = n * sizeof(float);

  std::vector<float> img(n);
  for (std::size_t i = 0; i < n; ++i) {
    img[i] = static_cast<float>(i % 256);
  }
  std::vector<float> out(n, 0.0f);

  const int radius = 2; // 5x5 window
  const double win = (2.0 * radius + 1) * (2.0 * radius + 1);

  auto time = [&](void (*fn)(const float *, float *, int, int)) {
    dojo::GpuTimer t;
    t.start();
    fn(img.data(), out.data(), w, h);
    return t.stop();
  };
  auto time_blur = [&](void (*fn)(const float *, float *, int, int, int)) {
    dojo::GpuTimer t;
    t.start();
    fn(img.data(), out.data(), w, h, radius);
    return t.stop();
  };

  // box blur: ~win adds + 1 divide per pixel; reads win floats per pixel.
  report("box_blur_naive", time_blur(dojo::level05::box_blur_naive), bytes,
         win + 1.0, win * 4.0, n, peak);
  report("box_blur_opt", time_blur(dojo::level05::box_blur_opt), bytes,
         win + 1.0, win * 4.0, n, peak);
  // sobel: ~20 FLOPs/pixel, reads 9 floats/pixel.
  report("sobel", time(dojo::level05::sobel), bytes, 20.0, 9.0 * 4.0, n, peak);

  std::puts("\nVerify on a CUDA machine:");
  std::puts("  nsys profile ./level05_demo        # timeline: compute vs copy");
  std::puts("  ncu --set full ./level05_demo      # per-kernel BW, occupancy, roofline dot");
  return 0;
}
