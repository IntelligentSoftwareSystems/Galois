class OnlineStat {
  unsigned int n;
  double mean;
  double M2;
  
public:
  OnlineStat() :n(0), mean(0.0), M2(0.0) {}

  void reset() {
    M2 = mean = 0.0;
    n = 0;
  }
  
  void insert(double x) {
    n += 1;
    double delta = x - mean;
    mean += delta / n;
    M2 += delta * (x - mean);
  }

  double getVariance() const {
    return M2/(n - 1);
  }

  double getStdDeviation() const {
    return M2/n;
  }

  unsigned int getCount() const {
    return n;
  }

  double getMean() const {
    return mean;
  }

};
