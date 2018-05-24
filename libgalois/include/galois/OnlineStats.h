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
  OnlineStat() :n(0), mean(0.0), M2(0.0), 
                _min(std::numeric_limits<double>::max()),
                _max(std::numeric_limits<double>::min())
  {}

  OnlineStat(const OnlineStat&) = default;

  void reset() {
    M2 = mean = 0.0;
    n = 0;
    _min = std::numeric_limits<double>::max();
    _max = std::numeric_limits<double>::min();
  }
  
  void insert(double x) {
    n += 1;
    double delta = x - mean;
    mean += delta / n;
    M2 += delta * (x - mean);
    _min = std::min(x, _min);
    _max = std::max(x, _max);
  }

  double getVariance() const { return M2/(n - 1); }
  double getStdDeviation() const { return M2/n; }
  unsigned int getCount() const { return n; }
  double getMean() const { return mean; }
  double getMin() const { return _min; }
  double getMax() const { return _max; }
  
  friend std::ostream& operator<<(std::ostream& os, const OnlineStat& s) {
    os << "{count " << s.getCount()
       << ", mean " << s.getMean()
       << ", stddev " << s.getStdDeviation()
       << ", var " << s.getVariance()
       << ", min " << s.getMin()
       << ", max " << s.getMax()
       << "}";
    return os;
  }
};

} //namespace galois

#endif
