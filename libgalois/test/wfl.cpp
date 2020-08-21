#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/WaterFallLock.h"
#include "galois/substrate/PerThreadStorage.h"

#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cxxabi.h>

unsigned iter       = 0;
unsigned numThreads = 0;

char bname[100];

template <typename T>
struct emp {
  galois::WaterFallLock<T>& w;
  std::vector<uint64_t>& arr_a;

  void operator()(const uint64_t& tid, const uint64_t& numThreads) {
    for (unsigned i = 0; i < iter; i++) {
      auto perm_tid = tid ? tid - 1 : numThreads - 1;
      arr_a[tid]++;
      w.template done<1>(tid);
      w.template wait<1>(perm_tid);
      arr_a[perm_tid]++;
      if (arr_a[perm_tid] != 2)
        std::abort();
      w.template done<2>(perm_tid);
      w.template wait<2>(tid);
      arr_a[tid]++;
      if (arr_a[tid] != 3)
        std::abort();
      w.template done<3>(tid);
      w.template wait<3>(perm_tid);
      arr_a[perm_tid] -= 3;
      if (arr_a[perm_tid] != 0)
        std::abort();
      w.template done<0>(perm_tid);
      w.template wait<0>(tid);
    }
  }
};

template <typename T>
void test(galois::WaterFallLock<T>* w, std::vector<uint64_t>& arr0) {
  if (w == nullptr) {
    std::cout << "skipping " << std::endl;
    return;
  }
  gethostname(bname, sizeof(bname));
  char* name = 0;
  int status;
  name = abi::__cxa_demangle(w->name(), 0, 0, &status);
  if (status || !name)
    std::abort();

  emp<T> e{*w, arr0};

  unsigned M = numThreads;
  while (M) {
    galois::setActiveThreads(M);
    w->reset();
    galois::Timer t;
    galois::on_each(e);
    t.start();
    galois::on_each(e);
    t.stop();
    std::cout << bname << "," << name << "," << M << "," << t.get_usec()
              << "\n";
    M -= 1;
  }
  free(name);
}

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

  using PTS = PerThreadStorage<unsigned>;
  using CLP = CacheLinePaddedArr<unsigned>;

  auto arr0 = std::vector<uint64_t>(numThreads, 0);
  test<PTS>(new WaterFallLock<PTS>(), arr0);
  test<CLP>(new WaterFallLock<CLP>(), arr0);
  return 0;
}
