#include "Galois/Launcher.h"
#include "Galois/Runtime/Timer.h"

// This is linux/bsd specific
#include <sys/time.h>

static bool firstRun = true;
static GaloisRuntime::Timer LaunchTimer;

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

GaloisRuntime::Timer::Timer()
  :_start_hi(0), _start_low(0), _stop_hi(0), _stop_low(0)
{}

void GaloisRuntime::Timer::start() {
  timeval start;
  gettimeofday(&start, 0);
  _start_hi = start.tv_sec;
  _start_low = start.tv_usec;
}

void GaloisRuntime::Timer::stop() {
  timeval stop;
  gettimeofday(&stop, 0);
  _stop_hi = stop.tv_sec;
  _stop_low = stop.tv_usec;
}

unsigned long GaloisRuntime::Timer::get() const {
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

unsigned long GaloisRuntime::Timer::get_usec() const {
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


GaloisRuntime::TimeAccumulator::TimeAccumulator()
  :ltimer(), acc(0)
{}

void GaloisRuntime::TimeAccumulator::start() {
  ltimer.start();
}

void GaloisRuntime::TimeAccumulator::stop() {
  ltimer.stop();
  acc += ltimer.get_usec();
}

unsigned long GaloisRuntime::TimeAccumulator::get() const {
  return acc / 1000;
}
