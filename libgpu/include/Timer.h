#pragma once
/*
   Timer.h

   Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#if !(_POSIX_TIMERS > 0)
#error "POSIX timers not available"
#endif

#ifdef _POSIX_MONOTONIC_CLOCK
#ifdef CLOCK_MONOTONIC_RAW
static clockid_t CLOCKTYPE    = CLOCK_MONOTONIC_RAW;
static const char* SCLOCKTYPE = "CLOCK_MONOTONIC_RAW";
#else
static clockid_t CLOCKTYPE = CLOCK_MONOTONIC static const char* SCLOCKTYPE =
    "CLOCK_MONOTONIC";
#endif /* CLOCK_MONOTONIC_RAW */
#else
#warning "CLOCK_MONOTONIC is unavailable, using CLOCK_REALTIME"
static clockid CLOCKTYPE      = CLOCK_REALTIME;
static const char* SCLOCKTYPE = "CLOCK_REALTIME";
#endif /* _POSIX_MONOTONIC_CLOCK */

#define NANOSEC 1000000000LL

namespace ggc {
class Timer {
  char const* name;
  struct timespec begin, end;
  bool active, valid;
  unsigned long long last;
  unsigned long long total;

public:
  Timer(const char* timer_name) {
    name   = timer_name;
    active = false;
    valid  = false;
    total  = 0;
  }

  unsigned long long normalize(const struct timespec& t) const {
    return t.tv_sec * NANOSEC + t.tv_nsec;
  }

  void reset() {
    assert(!active);
    total = 0;
    last  = 0;
  }

  void start() {
    assert(!active);
    active = true;
    valid  = false;
    if (clock_gettime(CLOCKTYPE, &begin) == -1) {
      if (errno == EINVAL) {
        fprintf(stderr, "%s (%d) not available.\n", SCLOCKTYPE, CLOCKTYPE);
        // exit?
      }
    }
  }

  void print() {
    printf("%s %llu %llu\n", name, normalize(begin), normalize(end));
  }
  void stop() {
    assert(active);

    if (clock_gettime(CLOCKTYPE, &end) == -1) {
      if (errno == EINVAL) {
        fprintf(stderr, "%s (%d) not available.\n", SCLOCKTYPE, CLOCKTYPE);
        // exit?
      }
    }

    // assert(normalize(end) > normalize(begin) // paranoid level 2

    last = normalize(end) - normalize(begin);
    total += last;
    active = false;
    valid  = true;
  }

  unsigned long long duration() const { return last; }

  unsigned long long duration_ms() const { return last * 1000 / NANOSEC; }

  unsigned long long duration_s() const { return last / NANOSEC; }

  unsigned long long total_duration() const { return total; }

  unsigned long long total_duration_ms() const {
    return total * 1000 / NANOSEC;
  }

  unsigned long long total_duration_s() const { return total / NANOSEC; }
};
} // namespace ggc

#if 0
__attribute__((constructor)) static void upgrade_timer(void) {
  struct timespec res;
  
// see if CLOCK_MONOTONIC_RAW is available at runtime
#if defined(_POSIX_MONOTONIC_CLOCK) && defined(__linux__)
    if(CLOCKTYPE == CLOCK_MONOTONIC) {
      int rv;
      clockid_t clockid;

#ifdef CLOCK_MONOTONIC_RAW
      clockid = CLOCK_MONOTONIC_RAW;
#else
      clockid = 4; // from bits/time.h
#endif

      rv = clock_getres(clockid, &res);
      if(rv == 0) {
	//fprintf(stderr, "Using CLOCK_MONOTONIC_RAW for Timer.\n");
	CLOCKTYPE = clockid;
	SCLOCKTYPE = "CLOCK_MONOTONIC_RAW";
      }
    }
#endif
}
#endif
