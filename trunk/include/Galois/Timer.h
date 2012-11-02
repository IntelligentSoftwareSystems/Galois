/** Simple timer support -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_TIMER_H
#define GALOIS_TIMER_H

namespace Galois {

  //! A simple timer
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

  //! A multi-start time accumulator.
  //! Gives the final runtime for a series of intervals
  class TimeAccumulator {
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
}
#endif

