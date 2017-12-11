#include "galois/Galois.h"
#include "galois/Timer.h"
#include "galois/runtime/Profile.h"


#include <iostream>


int main(int argc, char* argv[]) {

  galois::SharedMemSys G;


  auto numThreads = galois::setActiveThreads(std::stoul(argv[1]));

  size_t arrSz  = 1024*1024;
 
  size_t* arr = new size_t[arrSz];


  galois::runtime::profilePapi([&] (void) {
      galois::do_all(galois::iterate(0ul, arrSz),
        [&] (size_t i) {
          arr[i] = i;
        });
  },
  "arrayInit");

  size_t sum = 0;

  galois::runtime::profilePapi([&] (void) {
      galois::do_all(galois::iterate(0ul, arrSz),
        [&] (size_t i) {
          sum += arr[i];
        });
  },
  "arraySum");


  std::cout << "Array Sum = " << sum << std::endl;

  delete[] arr;

  return 0;
}
