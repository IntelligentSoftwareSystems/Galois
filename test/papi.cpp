#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/runtime/Profile.h"


#include <iostream>

template <typename V>
size_t vecSumSerial(V& vec) {
  galois::runtime::profilePapi([&] (void) {
    for(size_t i = 0, sz = vec.size(); i < sz; ++i) {
      vec[i] = i;
    }
  },
  "vecInit");

  size_t sum = 0;

  galois::runtime::profilePapi([&] (void) {
    for(size_t i = 0, sz = vec.size(); i < sz; ++i) {
      sum += vec[i];
    }
  },
  "vecSum");

  return sum;
}

template <typename V>
size_t vecSumParallel(V& vec) {
  galois::runtime::profilePapi([&] (void) {
      galois::do_all(galois::iterate(0ul, vec.size()),
        [&] (size_t i) {
          vec[i] = i;
        });
  },
  "vecInit");

  size_t sum = 0;

  galois::runtime::profilePapi([&] (void) {
      galois::do_all(galois::iterate(0ul, vec.size()),
        [&] (size_t i) {
          sum += vec[i];
        });
  },
  "vecSum");

  return sum;
}


int main(int argc, char* argv[]) {

  galois::SharedMemSys G;


  auto numThreads = galois::setActiveThreads(std::stoul(argv[1]));
  galois::runtime::reportParam("NULL", "Threads", numThreads);

  size_t vecSz  = 1024*1024;

  std::vector<size_t> vec(vecSz);
 
  size_t sum = vecSumSerial(vec);

  std::cout << "Array Sum = " << sum << std::endl;

  return 0;
}
