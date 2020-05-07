#pragma once

#include <random>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>
#ifdef GALOIS_USE_DIST
#include "deepgalois/gtypes.h"
#else
#include "deepgalois/types.h"
#endif

namespace deepgalois {

//! tracks max mem usage with rusage
// TODO use Galois's getrusage functionality
class ResourceManager {
public:
  ResourceManager() {}
  ~ResourceManager() {}
  // peak memory usage
  std::string get_peak_memory() {
    double kbm;
    struct rusage CurUsage;
    getrusage(RUSAGE_SELF, &CurUsage);
    kbm        = (double)CurUsage.ru_maxrss;
    double mbm = kbm / 1024.0;
    double gbm = mbm / 1024.0;
    return "Peak memory: " + to_string_with_precision(mbm, 3) + " MB; " +
           to_string_with_precision(gbm, 3) + " GB";
  }

private:
  template <typename T = double>
  std::string to_string_with_precision(const T a_value, const int& n) {
    std::ostringstream out;
    out << std::fixed;
    out << std::setprecision(n) << a_value;
    return out.str();
  }
};

// TODO don't need a separate timer: use Galois's regular timer
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

class random_generator {
public:
  static random_generator& get_instance() {
    static random_generator instance;
    return instance;
  }
  std::mt19937& operator()() { return gen_; }
  void set_seed(unsigned int seed) { gen_.seed(seed); }

private:
  random_generator() : gen_(1) {}
  std::mt19937 gen_;
};

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
uniform_rand(T min, T max) {
  std::uniform_int_distribution<T> dst(min, max);
  return dst(random_generator::get_instance()());
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
uniform_rand(T min, T max) {
  std::uniform_real_distribution<T> dst(min, max);
  return dst(random_generator::get_instance()());
}

// sequential prefix sum
template <typename InTy = unsigned, typename OutTy = unsigned>
inline std::vector<OutTy> prefix_sum(const std::vector<InTy>& in) {
  std::vector<OutTy> prefix(in.size() + 1);
  OutTy total = 0;
  for (size_t i = 0; i < in.size(); i++) {
    prefix[i] = total;
    total += (OutTy)in[i];
  }
  prefix[in.size()] = total;
  return prefix;
}

template <typename InTy = unsigned, typename OutTy = unsigned>
OutTy* parallel_prefix_sum(const std::vector<InTy>& in);

// Utility function to randomly select k items from [begin, end)
template <typename T = int>
inline T* select_k_items(T k, T begin, T end) {
  auto i = begin;

  // reservoir[] is the output array. Initialize
  // it with first k vertices
  T* reservoir = new T[k];
  for (; i < k; i++)
    reservoir[i] = i;

  // Use a different seed value so that we don't get
  // same result each time we run this program
  srand(time(NULL));

  // Iterate from the (k+1)th element to nth element
  for (; i < end; i++) {
    // Pick a random index from 0 to i.
    auto j = rand() % (i + 1);

    // If the randomly picked index is smaller than k,
    // then replace the element present at the index
    // with new element from stream
    if (j < k)
      reservoir[j] = i;
  }
  return reservoir;
}

// Utility function to find ceiling of r in arr[l..h]
template <typename T = int>
inline T find_ceil(T* arr, T r, T l, T h) {
  T mid;
  while (l < h) {
    mid = l + ((h - l) >> 1); // Same as mid = (l+h)/2
    (r > arr[mid]) ? (l = mid + 1) : (h = mid);
  }
  return (arr[l] >= r) ? l : -1;
}

// Utility function to select one element from n elements given a frequency
// (probability) distribution
// https://www.geeksforgeeks.org/random-number-generator-in-arbitrary-probability-distribution-fashion/
template <typename T = int>
T select_one_item(T n, T* dist) {
  T* offsets = new T[n];
  offsets[0] = dist[0];
  // compute the prefix sum of the distribution
  for (T i = 1; i < n; ++i)
    offsets[i] = offsets[i - 1] + dist[i];
  // offsets[n-1] is sum of all frequencies
  T sum = offsets[n - 1];
  T r   = (rand() % sum) + 1;
  // find which range r falls into, and return the index of the range
  return find_ceil(offsets, r, 0, n - 1);
}

acc_t masked_f1_score(size_t begin, size_t end, size_t count, mask_t* masks,
                      size_t num_classes, label_t* ground_truth, float_t* pred);

} // namespace deepgalois
