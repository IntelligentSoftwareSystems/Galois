#include "Galois/Launcher.h"
#include "Galois/Runtime/Timer.h"

// This is linux/bsd specific
#include <sys/time.h>

using namespace GaloisRuntime;

static bool firstRun = true;
static Timer LaunchTimer;

bool Galois::Launcher::isFirstRun() {
  return firstRun;
}

void Galois::Launcher::startTiming() {
  LaunchTimer.start();
}

void Galois::Launcher::stopTiming() {
  LaunchTimer.stop();
}

void Galois::Launcher::reset() {
  firstRun = false;
}

unsigned long Galois::Launcher::elapsedTime() {
  return LaunchTimer.get();
}

Timer::Timer()
  :_start_hi(0), _start_low(0), _stop_hi(0), _stop_low(0)
{}

void Timer::start() {
  timeval start;
  gettimeofday(&start, 0);
  _start_hi = start.tv_sec;
  _start_low = start.tv_usec;
}

void Timer::stop() {
  timeval stop;
  gettimeofday(&stop, 0);
  _stop_hi = stop.tv_sec;
  _stop_low = stop.tv_usec;
}

unsigned long Timer::get() const {
  unsigned long msec = _stop_hi - _start_hi;
  msec *= 1000;
  if (_stop_low > _start_low)
    msec += (_stop_low - _start_low) / 1000;
  else {
    msec -= 1000; //borrow
    msec += (_stop_low + 1000000 - _start_low) / 1000;
  }
  return msec;
}

unsigned long Timer::get_usec() const {
  unsigned long usec = _stop_hi - _start_hi;
  usec *= 1000000;
  if (_stop_low > _start_low)
    usec += (_stop_low - _start_low);
  else {
    usec -= 1000000; //borrow
    usec += (_stop_low + 1000000 - _start_low);
  }
  return usec;
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
