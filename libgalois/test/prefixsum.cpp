#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/WaterFallLock.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/PrefixSum.h"

#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cxxabi.h>
#include <utility>

unsigned iter       = 0;
unsigned numThreads = 0;

char bname[100];

template <typename T, typename Y>
void test(T& prefix_sum, uint64_t sz, Y* dst) {
  gethostname(bname, sizeof(bname));
  char* name = 0;
  int status;
  name = abi::__cxa_demangle(prefix_sum.name(), 0, 0, &status);
  if (status || !name)
    std::abort();

  auto run = [&prefix_sum, sz]() { prefix_sum.computePrefixSum(sz); };

  unsigned M = numThreads;
  while (M) {
    galois::setActiveThreads(M);
    galois::Timer t;
    run();
    for (uint64_t i = 0; i < sz; i++) {
      if (dst[i] != i + 1)
        std::abort();
    }

    t.start();
    run();
    t.stop();
    std::cout << bname << "," << name << "," << M << "," << t.get_usec()
              << "\n";
    M -= 1;
  }
  free(name);
}

uint64_t transmute(const std::pair<uint64_t, uint64_t>& p) {
  return p.second - p.first;
};
uint64_t scan_op(const std::pair<uint64_t, uint64_t>& p, const uint64_t& l) {
  return p.second - p.first + l;
};
uint64_t combiner(const uint64_t& f, const uint64_t& s) { return f + s; };

int main(int argc, char** argv) {
  galois::SharedMemSys Galois_runtime;
  if (argc > 1)
    iter = atoi(argv[1]);
  else
    iter = 16 * 1024;
  if (argc > 2)
    numThreads = atoi(argv[2]);
  else
    numThreads = galois::substrate::getThreadPool().getMaxThreads();

  gethostname(bname, sizeof(bname));
  using namespace galois;

  std::cout << "Host"
            << ","
            << "Lock Name"
            << ","
            << "numThreads"
            << ","
            << "Time (us)" << std::endl;

  // using PTS   = PerThreadStorage<unsigned>;

  auto src = (std::pair<uint64_t, uint64_t>*)malloc(
      sizeof(std::pair<uint64_t, uint64_t>) * (1 << 30));
  auto dst = (uint64_t*)malloc(sizeof(uint64_t) * (1 << 30));

  for (uint64_t i = 0; i < (1 << 30); i++)
    src[i] = {0, 1};

  using PSUM = PrefixSum<std::pair<uint64_t, uint64_t>, uint64_t, transmute,
                         scan_op, combiner, CacheLinePaddedArr>;

  PSUM prefix{src, dst};

  test<PSUM>(prefix, 50, dst);
  test<PSUM>(prefix, 1000, dst);
  test<PSUM>(prefix, 40000, dst);
  test<PSUM>(prefix, (1 << 30), dst);
  free(src);
  free(dst);

  return 0;
}
