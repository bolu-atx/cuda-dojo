#include "level02.cuh"

#include "dojo/cuda_utils.cuh"

#include <cstdio>
#include <vector>

// Level 2 demo: run each 2D-mapped kernel on a real-sized image and report the
// effective bandwidth. The transpose number is the interesting one — watch how
// far below the invert (pure streaming) bandwidth it lands once you implement
// it, and connect that gap to the strided-write trap.
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

  // invert — pure streaming: read all, write all. ~2x image bytes moved.
  {
    dojo::GpuTimer t;
    t.start();
    dojo::level02::invert(img.data(), out.data(), w, h, 255.0f);
    float ms = t.stop();
    std::printf("invert   %dx%d: %.3f ms, ~%.1f GB/s\n", w, h, ms,
                dojo::gbps(2 * bytes, ms));
  }

  // transpose — reads + writes the same number of bytes as invert, but one end
  // is strided. Same traffic, very different bandwidth: that's the lesson.
  {
    dojo::GpuTimer t;
    t.start();
    dojo::level02::transpose(img.data(), out.data(), w, h);
    float ms = t.stop();
    std::printf("transpose %dx%d: %.3f ms, ~%.1f GB/s (one end strided)\n", w, h,
                ms, dojo::gbps(2 * bytes, ms));
  }

  // crop — quarter-size window out of the center.
  {
    const int cw = w / 2, ch = h / 2;
    std::vector<float> cropped(static_cast<std::size_t>(cw) * ch, 0.0f);
    dojo::GpuTimer t;
    t.start();
    dojo::level02::crop(img.data(), w, h, cropped.data(), w / 4, h / 4, cw, ch);
    float ms = t.stop();
    std::printf("crop      %dx%d -> %dx%d: %.3f ms\n", w, h, cw, ch, ms);
  }

  return 0;
}
