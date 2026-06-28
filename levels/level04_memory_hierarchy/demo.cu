#include "level04.cuh"

#include "dojo/cuda_utils.cuh"

#include <cstdio>
#include <vector>

// Level 4 demo: A/B the baseline and the hierarchy-aware version of each
// problem. Predict before you run: tiled transpose should reach ~2x the naive
// bandwidth (it removes the strided global access); the constant/read-only blur
// helps most when the reuse isn't already being caught by L1.
int main() {
  dojo::print_device_info();

  const int w = 4096;
  const int h = 4096;
  const std::size_t n = static_cast<std::size_t>(w) * h;
  const std::size_t bytes = n * sizeof(float);

  std::vector<float> img(n);
  for (std::size_t i = 0; i < n; ++i) {
    img[i] = static_cast<float>(i % 256);
  }
  std::vector<float> out(n, 0.0f);

  auto time_transpose = [&](const char *name, void (*fn)(const float *, float *,
                                                         int, int)) {
    dojo::GpuTimer t;
    t.start();
    fn(img.data(), out.data(), w, h);
    float ms = t.stop();
    // Transpose moves the image twice (read + write).
    std::printf("%-18s %.3f ms, ~%.1f GB/s\n", name, ms, dojo::gbps(2 * bytes, ms));
  };

  std::puts("\n--- transpose: naive vs tiled (expect ~2x) ---");
  time_transpose("transpose_naive", dojo::level04::transpose_naive);
  time_transpose("transpose_tiled", dojo::level04::transpose_tiled);

  auto time_blur = [&](const char *name,
                       void (*fn)(const float *, float *, int, int)) {
    dojo::GpuTimer t;
    t.start();
    fn(img.data(), out.data(), w, h);
    float ms = t.stop();
    std::printf("%-18s %.3f ms, ~%.1f GB/s\n", name, ms, dojo::gbps(2 * bytes, ms));
  };

  std::puts("\n--- 3x3 blur: global vs constant/read-only ---");
  time_blur("blur3x3_global", dojo::level04::blur3x3_global);
  time_blur("blur3x3_constant", dojo::level04::blur3x3_constant);

  return 0;
}
