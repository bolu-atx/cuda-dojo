#include "level05.cuh"

#include "dojo/test.hpp"

#include <cmath>
#include <vector>

using namespace dojo::level05;

namespace {

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

float clamp_at(const std::vector<float> &in, int w, int h, int x, int y) {
  int nx = x < 0 ? 0 : (x >= w ? w - 1 : x);
  int ny = y < 0 ? 0 : (y >= h ? h - 1 : y);
  return in[static_cast<std::size_t>(ny) * w + nx];
}

std::vector<float> cpu_box_blur(const std::vector<float> &in, int w, int h,
                                int radius) {
  std::vector<float> out(in.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float sum = 0.0f;
      int count = 0;
      for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
          sum += clamp_at(in, w, h, x + dx, y + dy);
          ++count;
        }
      out[static_cast<std::size_t>(y) * w + x] = sum / static_cast<float>(count);
    }
  }
  return out;
}

std::vector<float> cpu_sobel(const std::vector<float> &in, int w, int h) {
  std::vector<float> out(in.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      auto a = [&](int dx, int dy) { return clamp_at(in, w, h, x + dx, y + dy); };
      float gx = -a(-1, -1) + a(1, -1) - 2 * a(-1, 0) + 2 * a(1, 0) - a(-1, 1) +
                 a(1, 1);
      float gy = -a(-1, -1) - 2 * a(0, -1) - a(1, -1) + a(-1, 1) + 2 * a(0, 1) +
                 a(1, 1);
      out[static_cast<std::size_t>(y) * w + x] = std::sqrt(gx * gx + gy * gy);
    }
  }
  return out;
}

} // namespace

// ---- box_blur_naive (baseline: should PASS) -------------------------------

DOJO_TEST_CASE(box_blur_naive_matches_host) {
  const int w = 1023, h = 577, r = 2;
  auto img = make_image(w, h);
  auto ref = cpu_box_blur(img, w, h, r);
  std::vector<float> out(img.size(), -1.0f);
  box_blur_naive(img.data(), out.data(), w, h, r);
  for (std::size_t i = 0; i < img.size(); ++i)
    DOJO_CHECK_CLOSE(out[i], ref[i], 1e-2);
}

// ---- box_blur_opt (stub: fails until implemented) -------------------------
// Must match the naive result exactly (to tolerance) — same spec, faster path.

DOJO_TEST_CASE(box_blur_opt_matches_naive) {
  const int w = 1023, h = 577, r = 2;
  auto img = make_image(w, h);
  auto ref = cpu_box_blur(img, w, h, r);
  std::vector<float> out(img.size(), -1.0f);
  box_blur_opt(img.data(), out.data(), w, h, r);
  for (std::size_t i = 0; i < img.size(); ++i)
    DOJO_CHECK_CLOSE(out[i], ref[i], 1e-2);
}

DOJO_TEST_CASE(box_blur_opt_radius1) {
  const int w = 200, h = 200, r = 1;
  auto img = make_image(w, h);
  auto ref = cpu_box_blur(img, w, h, r);
  std::vector<float> out(img.size(), -1.0f);
  box_blur_opt(img.data(), out.data(), w, h, r);
  for (std::size_t i = 0; i < img.size(); ++i)
    DOJO_CHECK_CLOSE(out[i], ref[i], 1e-2);
}

// ---- sobel (complete: should PASS) ----------------------------------------

DOJO_TEST_CASE(sobel_matches_host) {
  const int w = 513, h = 257;
  auto img = make_image(w, h);
  auto ref = cpu_sobel(img, w, h);
  std::vector<float> out(img.size(), -1.0f);
  sobel(img.data(), out.data(), w, h);
  for (std::size_t i = 0; i < img.size(); ++i)
    DOJO_CHECK_CLOSE(out[i], ref[i], 1e-2);
}

DOJO_TEST_MAIN()
