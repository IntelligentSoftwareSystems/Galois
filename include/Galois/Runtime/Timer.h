#ifndef TIMER_H_
#define TIMER_H_


namespace GaloisRuntime {

  //A simple timer
  class Timer {
    //This is so that implementations can vary without
    //forcing includes of target specific headers
    unsigned long _start_hi;
    unsigned long _start_low;
    unsigned long _stop_hi;
    unsigned long _stop_low;
  public:
    Timer();
    void start();
    void stop();
    unsigned long get() const;
    unsigned long get_usec() const;
  };

  //A multi-start time accumulator
  //Gives the final runtime for a series of intervals
  class TimeAccumulator {
    Timer ltimer;
    unsigned long acc;
  public:
    TimeAccumulator();
    void start();
    void stop(); //This adds the next timed interval to the total
    unsigned long get() const;
    TimeAccumulator& operator+=(const TimeAccumulator& rhs);
  };

}
#endif

