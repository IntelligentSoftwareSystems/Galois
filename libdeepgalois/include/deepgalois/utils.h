#pragma once

#include <random>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>

namespace deepgalois {

const std::string path =
    "/net/ohm/export/iss/inputs/Learning/"; // path to the input dataset

enum class net_phase { train, test };

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

inline bool bernoulli(float_t p) {
  return uniform_rand(float_t(0), float_t(1)) <= p;
}

//! Get masks from datafile where first line tells range of 
//! set to create mask from
inline size_t read_masks(std::string dataset_str, std::string mask_type,
                         size_t& begin, size_t& end,
                         std::vector<uint8_t>& masks) {
  if (dataset_str != "citeseer" && dataset_str != "cora" && dataset_str != "pubmed") {
    std::cout << "Dataset currently not supported\n";
    exit(1);
  }
  size_t i             = 0;
  size_t sample_count  = 0;
  std::string filename = path + dataset_str + "-" + mask_type + "_mask.txt";
  // std::cout << "Reading " << filename << "\n";
  std::ifstream in;
  std::string line;
  in.open(filename, std::ios::in);
  in >> begin >> end >> std::ws;
  while (std::getline(in, line)) {
    std::istringstream mask_stream(line);
    if (i >= begin && i < end) {
      unsigned mask = 0;
      mask_stream >> mask;
      if (mask == 1) {
        masks[i] = 1;
        sample_count++;
      }
    }
    i++;
  }
  std::cout << mask_type + "_mask range: [" << begin << ", " << end
    << ") Number of valid samples: " << sample_count << "\n";
  in.close();
  return sample_count;
}

}
