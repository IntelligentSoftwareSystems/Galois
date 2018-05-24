#include "galois/Galois.h"
#include <boost/iterator/counting_iterator.hpp>
#include <iostream>

//! [do_all example]
struct HelloWorld {
  void operator()(int i) const {
    std::cout << "Hello " << i << "\n";
  }
};

void helloWorld(int i) {
  std::cout << "Hello " << i << "\n";
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;

  if (argc < 3) {
    std::cerr << "<num threads> <num of iterations>\n";
    return 1;
  }
  unsigned int numThreads = atoi(argv[1]);
  int n = atoi(argv[2]);

  numThreads = galois::setActiveThreads(numThreads);
  std::cout << "Using " << numThreads << " threads and " << n << " iterations\n";

  std::cout << "Using a function object\n";
  galois::do_all(galois::iterate(boost::make_counting_iterator<int>(0), boost::make_counting_iterator<int>(n)), HelloWorld());

  std::cout << "Using a function pointer\n";
  galois::do_all(galois::iterate(boost::make_counting_iterator<int>(0), boost::make_counting_iterator<int>(n)), &helloWorld);

  std::cout << "Using a lambda\n";
  galois::do_all(galois::iterate(boost::make_counting_iterator<int>(0), boost::make_counting_iterator<int>(n)), [] (int i) { std::cout << "Hello " << i << "\n"; });
//! [do_all example]

  return 0;
}
