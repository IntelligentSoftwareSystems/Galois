#ifndef TIMER_H_
#define TIMER_H_
#include <sys/time.h>

class Timer {
public:
  Timer() {}
  void Start() { gettimeofday(&start_time_, NULL); }
  void Stop() {
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
  }
  double Seconds() const {
    return elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec / 1e6;
  }
  double Millisecs() const {
    return 1000 * elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec / 1000;
  }
  double Microsecs() const {
    return 1e6 * elapsed_time_.tv_sec + (double)elapsed_time_.tv_usec;
  }

private:
  struct timeval start_time_;
  struct timeval elapsed_time_;
};
#endif // TIMER_H_
