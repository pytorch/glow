#ifndef NOETHER_SUPPORT_H
#define NOETHER_SUPPORT_H

#include <chrono>
#include <iostream>

namespace noether {

/// A class for measuring the lifetime of some event
/// and for rate calculation.
class TimerGuard {
  int iterations_;
  std::chrono::time_point<std::chrono::system_clock> start;

public:
  TimerGuard(int iterations) : iterations_(iterations) {
    start = std::chrono::system_clock::now();
  }

  ~TimerGuard() {
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Rate: " << (iterations_ / elapsed_seconds.count())
              << "/sec\n";
  }
};

} // namespace noether

#endif // NOETHER_SUPPORT_H
