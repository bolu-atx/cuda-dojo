#include "level03.cuh"

#include "dojo/test.hpp"

#include <vector>

using namespace dojo::level03;

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

// CPU reference 3x3 box blur with border clamping — the spec both GPU blurs
// must match.
std::vector<float> cpu_blur3x3(const std::vector<float> &in, int w, int h) {
  std::vector<float> out(in.size());
  auto at = [&](int x, int y) {
    int nx = x < 0 ? 0 : (x >= w ? w - 1 : x);
    int ny = y < 0 ? 0 : (y >= h ? h - 1 : y);
    return in[static_cast<std::size_t>(ny) * w + nx];
  };
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float sum = 0.0f;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          sum += at(x + dx, y + dy);
      out[static_cast<std::size_t>(y) * w + x] = sum / 9.0f;
    }
  }
  return out;
}

void check_transpose(void (*fn)(const float *, float *, int, int), int w, int h) {
  auto img = make_image(w, h);
  std::vector<float> out(img.size(), -1.0f);
  fn(img.data(), out.data(), w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float in_v = img[static_cast<std::size_t>(y) * w + x];
      DOJO_CHECK_CLOSE(out[static_cast<std::size_t>(x) * h + y], in_v, 1e-3);
    }
  }
}

} // namespace

// ---- transpose_naive (baseline: should PASS) ------------------------------

DOJO_TEST_CASE(transpose_naive_matches_host) {
  check_transpose(transpose_naive, 257, 129); // non-square, non-multiple-of-tile
}

// ---- transpose_tiled (stub: fails until implemented) ----------------------

DOJO_TEST_CASE(transpose_tiled_matches_host) {
  check_transpose(transpose_tiled, 257, 129);
}

DOJO_TEST_CASE(transpose_tiled_partial_tile) {
  // 33x33 forces every edge tile to be partial — exercises both bounds guards.
  check_transpose(transpose_tiled, 33, 33);
}

// ---- blur3x3_global (baseline: should PASS) -------------------------------

DOJO_TEST_CASE(blur3x3_global_matches_host) {
  const int w = 1023, h = 577;
  auto img = make_image(w, h);
  auto ref = cpu_blur3x3(img, w, h);
  std::vector<float> out(img.size(), -1.0f);
  blur3x3_global(img.data(), out.data(), w, h);
  for (std::size_t i = 0; i < img.size(); ++i) {
    DOJO_CHECK_CLOSE(out[i], ref[i], 1e-3);
  }
}

// ---- blur3x3_constant (stub: fails until implemented) ---------------------

DOJO_TEST_CASE(blur3x3_constant_matches_host) {
  const int w = 1023, h = 577;
  auto img = make_image(w, h);
  auto ref = cpu_blur3x3(img, w, h);
  std::vector<float> out(img.size(), -1.0f);
  blur3x3_constant(img.data(), out.data(), w, h);
  for (std::size_t i = 0; i < img.size(); ++i) {
    DOJO_CHECK_CLOSE(out[i], ref[i], 1e-3);
  }
}

DOJO_TEST_MAIN()
