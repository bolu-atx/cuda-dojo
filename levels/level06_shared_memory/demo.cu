#include "level06.cuh"

#include "dojo/cuda_utils.cuh"

#include <cstdio>
#include <vector>

// Level 6 demo and profiling target. Runs the shared-memory box filter and the
// separable blur on a large image and reports effective bandwidth.
//
// What to look for under the profilers:
//   ncu --set full ./level06_demo
//   - "Shared Memory Bank Conflicts" should be ~0 for box_filter_shared thanks
//     to the +1 inner-dimension pad. Drop the pad and watch it spike.
//   - The box filter is memory-bound: compare achieved GB/s against your GPU's
//     peak. The separable version moves ~2*(2r+1)/(2r+1)^2 of the traffic, so
//     once you implement it, it should land closer to peak (fewer bytes/pixel).
int main() {
  dojo::print_device_info();

  const int w = 4096, h = 4096;
  const int radius = 4; // (2*4+1)^2 = 81-tap box; separable does 2*9 = 18 reads
  std::printf("\nimage = %dx%d (%.1f MB), box radius = %d\n", w, h,
              w * h * sizeof(float) / (1024.0 * 1024.0), radius);

  std::vector<float> img(static_cast<std::size_t>(w) * h), out(img.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      img[static_cast<std::size_t>(y) * w + x] =
          static_cast<float>((x * 7 + y * 13) % 251);
    }
  }

  // Box filter reads input + writes output once each at the global level (the
  // shared tile absorbs the neighbor reuse), so ~2 image-passes of traffic.
  const std::size_t two_passes = 2ull * img.size() * sizeof(float);

  // ---- shared-memory box filter (worked example) ----
  {
    dojo::GpuTimer t;
    t.start();
    dojo::level06::box_filter_shared(img.data(), out.data(), w, h, radius);
    float ms = t.stop();
    std::printf("\nbox_filter_shared: out[100,100] = %.3f, %.3f ms, ~%.1f GB/s\n",
                out[100 * w + 100], ms, dojo::gbps(two_passes, ms));
  }

  // ---- separable blur (stub: numbers are meaningless until implemented) ----
  {
    dojo::GpuTimer t;
    t.start();
    dojo::level06::separable_blur(img.data(), out.data(), w, h, radius);
    float ms = t.stop();
    // Separable touches the image ~3x (in->tmp, tmp->out reads/writes).
    std::printf("separable_blur:    out[100,100] = %.3f, %.3f ms (stub: expect 0)\n",
                out[100 * w + 100], ms);
  }

  return 0;
}
