// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef TIMER_H_
#define TIMER_H_

#include <sys/time.h>


/*
GAP Benchmark Suite
Class:  Timer
Author: Scott Beamer

Simple timer that wraps gettimeofday
*/


class Timer {
 public:
  Timer() {}

  void Start() {
    gettimeofday(&start_time_, NULL);
  }

  void Stop() {
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
  }

  double Seconds() const {
    return elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1e6;
  }

  double Millisecs() const {
    return 1000*elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec/1000;
  }

  double Microsecs() const {
    return 1e6*elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec;
  }

 private:
  struct timeval start_time_;
  struct timeval elapsed_time_;
};

// Times op's execution using the timer t
#define TIME_OP(t, op) { t.Start(); (op); t.Stop(); }

#endif  // TIMER_H_
