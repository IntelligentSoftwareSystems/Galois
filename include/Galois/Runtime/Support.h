/** Reporting and utility code -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_SUPPORT_H
#define GALOIS_RUNTIME_SUPPORT_H

#include <cmath>
#include <algorithm>
#include <limits>
#include <ostream>

// Hack to get full times out easily
//#define __HAS_DUMP 1

#ifdef __HAS_DUMP
#include <vector>
#include <fstream>
#endif

namespace GaloisRuntime {
#if __HAS_DUMP
  static int dump_counter = 0;
#endif

//! Compute mean and variance online
class OnlineStatistics {
  double m_min;
  double m_max;
  double m_mean;
  double m_m2;
  size_t m_n;
#ifdef __HAS_DUMP
  std::vector<double> m_values;
#endif
public:
  OnlineStatistics():
    m_min(std::numeric_limits<double>::max()),
    m_max(std::numeric_limits<double>::min()),
    m_mean(0),
    m_m2(0),
    m_n(0) { }

  template<typename T> void push(T x) {
#ifdef __HAS_DUMP
    m_values.push_back(x);
#endif
    m_min = std::min(m_min, static_cast<double>(x));
    m_max = std::max(m_max, static_cast<double>(x));
    ++m_n;
    double delta = x - m_mean;
    m_mean += delta/m_n;
    m_m2 += delta*(x-m_mean);
  }

  void dump(unsigned int tid) const {
#ifdef __HAS_DUMP
    char name[100];
    snprintf(name, 100, "dump%d.%d.txt", tid, dump_counter);
    std::ofstream out(name);
   
    for (std::vector<double>::const_iterator ii = m_values.begin(), ei = m_values.end(); ii != ei; ++ii) {
      out << *ii << "\n";
    }
    dump_counter++;
#endif
  }

  double mean() const { return m_mean; }
  double sample_variance() const { return m_n > 1 ? m_m2 / (m_n - 1) : 0; }
  double variance() const { return m_n > 0 ? m_m2 / m_n : 0; }
  double min() const { return m_min; }
  double max() const { return m_max; }
  size_t n() const { return m_n; }
};

extern bool inGaloisForEach;

std::ostream& operator<<(std::ostream& os, const OnlineStatistics& x);

//Report Statistics
void reportStatSum(const char* text, unsigned long val, const char* loopname = 0);

void reportStatAvg(const char* text, unsigned long val, const char* loopname = 0);

//Done with one loop and trigger list summary
void statDone();

void reportFlush();
}

#endif

