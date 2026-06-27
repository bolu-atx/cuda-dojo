#pragma once

// ---------------------------------------------------------------------------
// dojo micro test harness — zero dependencies, ~1 screen of code.
//
//   #include "dojo/test.hpp"
//
//   DOJO_TEST_CASE(vector_add_matches_host) {
//     DOJO_CHECK(1 + 1 == 2);
//     DOJO_CHECK_CLOSE(3.14, 3.1, 0.1);
//   }
//
//   DOJO_TEST_MAIN()   // exactly once per test executable
//
// A failing CHECK records the failure but keeps going, so one run reports all
// problems. main() returns non-zero if any case failed, which is what CTest
// keys off of.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace dojo::test {

struct Case {
  std::string name;
  std::function<void()> fn;
};

inline std::vector<Case> &registry() {
  static std::vector<Case> r;
  return r;
}

inline int &failures() {
  static int f = 0;
  return f;
}

struct Registrar {
  Registrar(std::string name, std::function<void()> fn) {
    registry().push_back({std::move(name), std::move(fn)});
  }
};

inline int run_all() {
  int failed_cases = 0;
  for (const auto &c : registry()) {
    const int before = failures();
    std::printf("[ RUN  ] %s\n", c.name.c_str());
    c.fn();
    if (failures() > before) {
      std::printf("[ FAIL ] %s\n", c.name.c_str());
      ++failed_cases;
    } else {
      std::printf("[  OK  ] %s\n", c.name.c_str());
    }
  }
  std::printf("\n%zu case(s), %d failed\n", registry().size(), failed_cases);
  return failed_cases == 0 ? 0 : 1;
}

} // namespace dojo::test

#define DOJO_TEST_CASE(name)                                                    \
  static void name();                                                           \
  static ::dojo::test::Registrar registrar_##name(#name, name);                 \
  static void name()

#define DOJO_CHECK(cond)                                                        \
  do {                                                                          \
    if (!(cond)) {                                                              \
      ++::dojo::test::failures();                                               \
      std::printf("  CHECK failed: %s  (%s:%d)\n", #cond, __FILE__, __LINE__);  \
    }                                                                           \
  } while (0)

#define DOJO_CHECK_CLOSE(a, b, eps)                                             \
  do {                                                                          \
    double da_ = static_cast<double>(a);                                        \
    double db_ = static_cast<double>(b);                                        \
    double diff_ = std::fabs(da_ - db_);                                        \
    if (diff_ > (eps)) {                                                        \
      ++::dojo::test::failures();                                               \
      std::printf("  CHECK_CLOSE failed: |%g - %g| = %g > %g  (%s:%d)\n", da_,  \
                  db_, diff_, static_cast<double>(eps), __FILE__, __LINE__);    \
    }                                                                           \
  } while (0)

#define DOJO_TEST_MAIN()                                                        \
  int main() { return ::dojo::test::run_all(); }
