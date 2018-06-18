/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_ONLINESTATS_H
#define GALOIS_ONLINESTATS_H

#include <limits>
#include <ostream>

namespace galois {

class OnlineStat {
  unsigned int n;
  double mean;
  double M2;
  double _min;
  double _max;

public:
  OnlineStat()
      : n(0), mean(0.0), M2(0.0), _min(std::numeric_limits<double>::max()),
        _max(std::numeric_limits<double>::min()) {}

  OnlineStat(const OnlineStat&) = default;

  void reset() {
    M2 = mean = 0.0;
    n         = 0;
    _min      = std::numeric_limits<double>::max();
    _max      = std::numeric_limits<double>::min();
  }

  void insert(double x) {
    n += 1;
    double delta = x - mean;
    mean += delta / n;
    M2 += delta * (x - mean);
    _min = std::min(x, _min);
    _max = std::max(x, _max);
  }

  double getVariance() const { return M2 / (n - 1); }
  double getStdDeviation() const { return M2 / n; }
  unsigned int getCount() const { return n; }
  double getMean() const { return mean; }
  double getMin() const { return _min; }
  double getMax() const { return _max; }

  friend std::ostream& operator<<(std::ostream& os, const OnlineStat& s) {
    os << "{count " << s.getCount() << ", mean " << s.getMean() << ", stddev "
       << s.getStdDeviation() << ", var " << s.getVariance() << ", min "
       << s.getMin() << ", max " << s.getMax() << "}";
    return os;
  }
};

} // namespace galois

#endif
