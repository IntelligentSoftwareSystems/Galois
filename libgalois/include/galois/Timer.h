#ifndef GALOIS_TIMER_H
#define GALOIS_TIMER_H

#include "galois/runtime/Statistics.h"

#include "boost/utility.hpp"

#include <chrono>
#include <limits>

#include <cstdio>
#include <ctime>

namespace galois {

//! Flag type for {@link StatTimer}
struct start_now_t {};
constexpr start_now_t start_now = start_now_t();

//! A simple timer
class Timer: private boost::noncopyable {
  typedef std::chrono::steady_clock clockTy;
  //typedef std::chrono::high_resolution_clock clockTy;
  std::chrono::time_point<clockTy> startT, stopT;
public:
  Timer() = default;
  Timer(const start_now_t) { start(); }
  void start();
  void stop();
  unsigned long get() const;
  unsigned long get_usec() const;
};

//! A multi-start time accumulator.
//! Gives the final runtime for a series of intervals
class TimeAccumulator: private boost::noncopyable {
  Timer ltimer;
  unsigned long acc;
public:
  TimeAccumulator();
  void start();
  //!adds the current timed interval to the total
  void stop();
  unsigned long get() const;
  TimeAccumulator& operator+=(const TimeAccumulator& rhs);
  TimeAccumulator& operator+=(const Timer& rhs);
};

//! Galois Timer that automatically reports stats upon destruction
//! Provides statistic interface around timer
class StatTimer : public TimeAccumulator {
  gstl::Str name;
  gstl::Str region;
  bool valid;

protected:
  void init(const char* n, const char* r, bool s) {
    n = n ? n : "Time";
    r = r? r : "(NULL)";

    name = gstl::makeStr(n);
    region = gstl::makeStr(r);

    valid = false;
    if (s)
      start();
  }

public:
  StatTimer(const char* const n) { init(n, nullptr, false); }
  StatTimer(const char* const n, start_now_t t) { init(n, nullptr, true); }

  StatTimer(const char* const n, const char* const r) { init(n, r, false); }
  StatTimer(const char* const n, const char* const r, start_now_t t) { init(n, r, true); }

  StatTimer() { init(nullptr, nullptr, false); }
  StatTimer(start_now_t t) { init(nullptr, nullptr, true); }

  ~StatTimer() {
    if (valid)
      stop();
    if (TimeAccumulator::get()) // only report non-zero stat
      galois::runtime::reportStat_Tmax(region, name, TimeAccumulator::get());
  }

  void start() {
    TimeAccumulator::start();
    valid = true;
  }

  void stop() {
    valid = false;
    TimeAccumulator::stop();
  }
};

template <bool Enable>
class CondStatTimer: public StatTimer {
public:
  CondStatTimer(const char* region): StatTimer("Time", region) {}
};

template <> class CondStatTimer<false> {
public:

  CondStatTimer(const char* name) {}

  void start(void) const {}

  void stop(void) const {}
};

template <typename F>
void timeThis(const F& f, const char* const name) {
  StatTimer t("Time", name);

  t.start();

  f();

  t.stop();
}

template <bool enabled>
class ThreadTimer: private boost::noncopyable {
  timespec m_start;
  timespec m_stop;
  int64_t  m_nsec;

public:
  ThreadTimer (): m_nsec (0) {};

  void start (void) {
    clock_gettime (CLOCK_THREAD_CPUTIME_ID, &m_start);
  }

  void stop (void) {
    clock_gettime (CLOCK_THREAD_CPUTIME_ID, &m_stop);
    m_nsec += (m_stop.tv_nsec - m_start.tv_nsec);
    m_nsec += ((m_stop.tv_sec - m_start.tv_sec) << 30); // multiply by 1G
  }

  int64_t get_nsec(void) const { return m_nsec; }

  int64_t get_sec(void) const { return (m_nsec >> 30); }

  int64_t get_msec(void) const { return (m_nsec >> 20); }

};

template <>
class ThreadTimer<false> {
public:
  void start (void) const  {}
  void stop (void) const  {}
  int64_t get_nsec (void) const { return 0; }
  int64_t get_sec (void) const  { return 0; }
  int64_t get_msec (void) const  { return 0; }
};



template <bool enabled>
class PerThreadTimer: private boost::noncopyable {

protected:

  const char* const region;
  const char* const category;

  substrate::PerThreadStorage<ThreadTimer<enabled> > timers;


  void reportTimes(void) {

    int64_t minTime = std::numeric_limits<int64_t>::max();


    for (unsigned i = 0; i < timers.size(); ++i) {

      auto ns = timers.getRemote(i)->get_nsec();

      minTime = std::min(minTime, ns);
    }

    std::string timeCat = category + std::string("-per-thread-times(ns)");
    std::string lagCat = category + std::string("-per-thread-lag(ns)");

    galois::substrate::getThreadPool(galois::getActiveThreads(),
        [&] (void) {
          auto ns = timers.getLocal()->get_nsec();
          auto lag = ns - minTime;
          assert(lag > 0 && "negative time lag from min is impossible");

          galois::runtime::reportStat_Tmax(region, timeCat.c_str(), ns);
          galois::runtime::reportStat_Tmax(region, lagCat.c_str(), lag);
        });

    // for (unsigned i = 0; i < timers.size(); ++i) {
//
      // auto ns = timers.getRemote(i)->get_nsec();
      // auto lag = ns - minTime;
      // assert(lag > 0 && "negative time lag from min is impossible");
//
      // galois::runtime::reportStat(region, lagCat.c_str(), lag, i);
      // galois::runtime::reportStat(region, timeCat.c_str(), ns, i);
    // }
  }

public:

  explicit PerThreadTimer(const char* const _region, const char* const _category)
    :
      region(_region),
      category(_category)
  {}

  ~PerThreadTimer(void) {
    reportTimes();
  }

  void start(void) {
    timers.getLocal()->start();
  }

  void stop(void) {
    timers.getLocal()->stop();
  }
};

template<> class PerThreadTimer<false> {

public:
  explicit PerThreadTimer(const char* const _region, const char* const _category)
  {}

  void start(void) const {}

  void stop(void) const {}

};

} // end namespace galois
#endif
