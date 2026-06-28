#include "level02.cuh"

#include "dojo/test.hpp"

#include <vector>

using namespace dojo::level02;

namespace {

// Deterministic test image: value depends on (x, y) so transpose/crop bugs that
// swap or offset coordinates actually change the result.
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

} // namespace

// ---- invert (complete worked example: should PASS) ------------------------

DOJO_TEST_CASE(invert_matches_host) {
  // Odd, non-multiple-of-block dimensions stress the two-sided boundary guard.
  const int w = 1023, h = 577;
  auto img = make_image(w, h);
  std::vector<float> out(img.size(), -1.0f);

  invert(img.data(), out.data(), w, h, 255.0f);

  for (std::size_t i = 0; i < img.size(); ++i) {
    DOJO_CHECK_CLOSE(out[i], 255.0f - img[i], 1e-3);
  }
}

DOJO_TEST_CASE(invert_handles_1x1) {
  const int w = 1, h = 1;
  std::vector<float> img = {42.0f};
  std::vector<float> out = {-1.0f};
  invert(img.data(), out.data(), w, h, 255.0f);
  DOJO_CHECK_CLOSE(out[0], 213.0f, 1e-3);
}

// ---- crop (stub: fails until implemented) ---------------------------------

DOJO_TEST_CASE(crop_matches_host) {
  const int inW = 640, inH = 480;
  const int x0 = 37, y0 = 51; // deliberately not block-aligned
  const int outW = 200, outH = 150;
  auto img = make_image(inW, inH);
  std::vector<float> out(static_cast<std::size_t>(outW) * outH, -1.0f);

  crop(img.data(), inW, inH, out.data(), x0, y0, outW, outH);

  for (int y = 0; y < outH; ++y) {
    for (int x = 0; x < outW; ++x) {
      float expected = img[static_cast<std::size_t>(y + y0) * inW + (x + x0)];
      DOJO_CHECK_CLOSE(out[static_cast<std::size_t>(y) * outW + x], expected, 1e-3);
    }
  }
}

// ---- transpose (stub: fails until implemented) ----------------------------

DOJO_TEST_CASE(transpose_matches_host) {
  // Non-square + non-multiple-of-block so a (w,h) vs (h,w) mix-up is caught.
  const int w = 257, h = 129;
  auto img = make_image(w, h);
  std::vector<float> out(img.size(), -1.0f);

  transpose(img.data(), out.data(), w, h);

  // out is h(rows) x w(cols): out[x*h + y] == in[y*w + x].
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float in_v = img[static_cast<std::size_t>(y) * w + x];
      DOJO_CHECK_CLOSE(out[static_cast<std::size_t>(x) * h + y], in_v, 1e-3);
    }
  }
}

DOJO_TEST_CASE(transpose_single_block) {
  // Fits in one 32x8 block — isolates the index math from grid tiling.
  const int w = 16, h = 8;
  auto img = make_image(w, h);
  std::vector<float> out(img.size(), -1.0f);

  transpose(img.data(), out.data(), w, h);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float in_v = img[static_cast<std::size_t>(y) * w + x];
      DOJO_CHECK_CLOSE(out[static_cast<std::size_t>(x) * h + y], in_v, 1e-3);
    }
  }
}

DOJO_TEST_MAIN()
