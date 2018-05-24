#include "galois/Timer.h"

using namespace galois;

void Timer::start() {
  startT = clockTy::now();
}

void Timer::stop() {
  stopT = clockTy::now();
}

unsigned long Timer::get() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(stopT-startT).count();
}

unsigned long Timer::get_usec() const {
  return std::chrono::duration_cast<std::chrono::microseconds>(stopT-startT).count();
}


TimeAccumulator::TimeAccumulator()
  :ltimer(), acc(0)
{}

void TimeAccumulator::start() {
  ltimer.start();
}

void TimeAccumulator::stop() {
  ltimer.stop();
  acc += ltimer.get_usec();
}

unsigned long TimeAccumulator::get() const {
  return acc / 1000;
}

TimeAccumulator& TimeAccumulator::operator+=(const TimeAccumulator& rhs) {
  acc += rhs.acc;
  return *this;
}

TimeAccumulator& TimeAccumulator::operator+=(const Timer& rhs) {
  acc += rhs.get_usec();
  return *this;
}
