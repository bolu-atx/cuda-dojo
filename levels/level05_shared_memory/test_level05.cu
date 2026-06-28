#include "level05.cuh"

#include "dojo/test.hpp"

#include <algorithm>
#include <vector>

using namespace dojo::level05;

namespace {

// Deterministic image: value depends on (x, y) so a halo/offset bug — reading
// the wrong neighbor or losing a border column — actually changes the output.
std::vector<float> make_image(int w, int h) {
  std::vector<float> img(static_cast<std::size_t>(w) * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      img[static_cast<std::size_t>(y) * w + x] =
          static_cast<float>((x * 7 + y * 13) % 251);
    }
  }
  return img;
}

// Plain CPU reference: (2r+1)x(2r+1) box mean with coordinates clamped to the
// border, dividing by the full window count. Both the shared and separable GPU
// kernels must reproduce this exactly (the separable clamp is per-axis, which
// composes to this 2D clamp).
std::vector<float> cpu_box(const std::vector<float> &in, int w, int h,
                           int radius) {
  std::vector<float> out(in.size());
  const float count = static_cast<float>((2 * radius + 1) * (2 * radius + 1));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float sum = 0.0f;
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          int nx = std::min(std::max(x + dx, 0), w - 1);
          int ny = std::min(std::max(y + dy, 0), h - 1);
          sum += in[static_cast<std::size_t>(ny) * w + nx];
        }
      }
      out[static_cast<std::size_t>(y) * w + x] = sum / count;
    }
  }
  return out;
}

void check_equal(const std::vector<float> &got, const std::vector<float> &ref) {
  for (std::size_t i = 0; i < ref.size(); ++i) {
    DOJO_CHECK_CLOSE(got[i], ref[i], 1e-3);
  }
}

} // namespace

// ---- box_filter_shared (complete worked example: should PASS) -------------

DOJO_TEST_CASE(box_filter_matches_host) {
  // Odd, non-multiple-of-16 dimensions stress the two-sided guard and the halo
  // load on partial edge tiles.
  const int w = 1023, h = 577, radius = 3;
  auto img = make_image(w, h);
  std::vector<float> out(img.size(), -1.0f);

  box_filter_shared(img.data(), out.data(), w, h, radius);
  check_equal(out, cpu_box(img, w, h, radius));
}

DOJO_TEST_CASE(box_filter_handles_1x1) {
  // Every neighbor clamps onto the single pixel, so the mean is that pixel.
  const int w = 1, h = 1, radius = 2;
  std::vector<float> img = {42.0f};
  std::vector<float> out = {-1.0f};
  box_filter_shared(img.data(), out.data(), w, h, radius);
  DOJO_CHECK_CLOSE(out[0], 42.0f, 1e-3);
}

DOJO_TEST_CASE(box_filter_single_block) {
  // Smaller than one 16x16 block: isolates index math from grid tiling.
  const int w = 12, h = 10, radius = 1;
  auto img = make_image(w, h);
  std::vector<float> out(img.size(), -1.0f);
  box_filter_shared(img.data(), out.data(), w, h, radius);
  check_equal(out, cpu_box(img, w, h, radius));
}

DOJO_TEST_CASE(box_filter_radius_exceeds_extent) {
  // radius (7) larger than the whole image (5x5): the halo is almost entirely
  // clamped border. Catches off-by-one clamping in the cooperative load.
  const int w = 5, h = 5, radius = 7;
  auto img = make_image(w, h);
  std::vector<float> out(img.size(), -1.0f);
  box_filter_shared(img.data(), out.data(), w, h, radius);
  check_equal(out, cpu_box(img, w, h, radius));
}

// ---- separable_blur (stub: fails until implemented) -----------------------

DOJO_TEST_CASE(separable_matches_host) {
  // Non-square + non-multiple-of-block so a (w,h) vs (h,w) mix-up is caught.
  const int w = 257, h = 129, radius = 4;
  auto img = make_image(w, h);
  std::vector<float> out(img.size(), -1.0f);

  separable_blur(img.data(), out.data(), w, h, radius);
  check_equal(out, cpu_box(img, w, h, radius));
}

DOJO_TEST_CASE(separable_single_block) {
  const int w = 16, h = 16, radius = 2;
  auto img = make_image(w, h);
  std::vector<float> out(img.size(), -1.0f);
  separable_blur(img.data(), out.data(), w, h, radius);
  check_equal(out, cpu_box(img, w, h, radius));
}

DOJO_TEST_MAIN()
