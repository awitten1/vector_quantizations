#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << ": FAILED: " #cond "\n"; \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                               \
  do {                                                                      \
    auto va_ = (a);                                                         \
    auto vb_ = (b);                                                         \
    if (std::abs(va_ - vb_) > (tol)) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << ": FAILED: " #a " ~= " #b \
                << " (" << va_ << " vs " << vb_ << ")\n";                   \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                       \
  } while (0)

#define CHECK_THROWS(expr, Exc)  \
  do {                           \
    bool threw_ = false;         \
    try {                        \
      (void)(expr);              \
    } catch (const Exc&) {       \
      threw_ = true;             \
    }                            \
    CHECK(threw_ && #expr);      \
  } while (0)
