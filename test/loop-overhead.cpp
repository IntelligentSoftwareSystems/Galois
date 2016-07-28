#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Substrate/ThreadPool.h"

#include <iostream>
#include <cstdlib>
#include <omp.h>

int RandomNumber () { return (rand()%1000000); }
unsigned iter = 1;

struct emp {
  template<typename T>
  void operator()(const T& t) const { 
    Galois::Substrate::compilerBarrier(); 
  }
  template<typename T, typename C>
  void operator()(const T& t, const C& c) const { Galois::Substrate::compilerBarrier(); }
  typedef int tt_does_not_need_push;
  typedef int tt_does_not_need_stats;
  typedef int tt_does_not_need_aborts;
};

unsigned t_inline(std::vector<unsigned>& V, unsigned num) {
  Galois::Timer t;
  t.start();
  emp e;
  for (unsigned x = 0; x < iter; ++x)
    for (unsigned i = 0; i < num; ++i)
      e(i);
  t.stop();
  return t.get();
}

unsigned t_stl(std::vector<unsigned>& V, unsigned num) {
  Galois::Timer t;
  t.start();
  for (unsigned x = 0; x < iter; ++x)
    std::for_each(V.begin(), V.begin()+num, emp());
  t.stop();
  return t.get();
}

unsigned t_omp(std::vector<unsigned>& V, unsigned num, unsigned th) {
  omp_set_num_threads(th); //Galois::Runtime::LL::getMaxThreads());
   
  Galois::Timer t;
  t.start();
  for (unsigned x = 0; x < iter; ++x) {
    emp f;
#pragma omp parallel for schedule(guided)
    for (unsigned n = 0; n < num; ++n)
      f(n);
  }
  t.stop();
  return t.get();
}

unsigned t_doall(bool burn, bool steal, std::vector<unsigned>& V, unsigned num, unsigned th) {
  Galois::setActiveThreads(th); //Galois::Runtime::LL::getMaxThreads());
  if (burn)
    Galois::Substrate::ThreadPool::getThreadPool().burnPower(th);
   
  Galois::Timer t;
  t.start();
  for (unsigned x = 0; x < iter; ++x)
    Galois::do_all(V.begin(), V.begin()+num, emp(), Galois::do_all_steal<>());
  t.stop();
  return t.get();
}

unsigned t_foreach(bool burn, std::vector<unsigned>& V, unsigned num, unsigned th) {
  Galois::setActiveThreads(th);
  if (burn)
    Galois::Substrate::ThreadPool::getThreadPool().burnPower(th);
  
  Galois::Timer t;
  t.start();
  for (unsigned x = 0; x < iter; ++x)
    Galois::for_each(V.begin(), V.begin() + num, emp(), Galois::wl<Galois::WorkList::StableIterator<>>());
  t.stop();
  return t.get();
}

void test(std::string header, unsigned maxThreads, unsigned minVec, unsigned maxVec,
          std::function<unsigned(std::vector<unsigned>&, unsigned, unsigned)> func) {
  std::cout << header << "";
  for (unsigned M = maxThreads; M; M >>= 1)
    std::cout << ",\t" << M;
  std::cout << "\n";
  std::vector<unsigned> V(maxVec);
  for (unsigned v = minVec; v < maxVec; v <<= 2) {
    std::cout << v << "\t";
    for (unsigned M = maxThreads; M; M >>= 1) {
      std::cout << ",\t" << func(V, v, M);
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main(int argc, char** argv) {
  using namespace std::placeholders;
#pragma omp parallel for
  for (int x = 0; x< 100; ++x)
    {}

  
  unsigned maxVector = 16;
  if (argc > 1)
    iter = atoi(argv[1]);
  if (!iter)
    iter = 16*1024;
  if (argc > 2)
    maxVector = atoi(argv[2]);
  if (!maxVector)
    maxVector = 1024*1024;

  unsigned M = Galois::Substrate::ThreadPool::getThreadPool().getMaxThreads() / 2;
  test("inline\t",  1, 16, maxVector, [] (std::vector<unsigned>& V, unsigned num, unsigned th) { return t_inline(V, num); });
  test("stl\t",     1, 16, maxVector, [] (std::vector<unsigned>& V, unsigned num, unsigned th) { return t_stl(V, num); });
  test("omp\t",     M, 16, maxVector, t_omp);
  test("doall N W", M, 16, maxVector, std::bind(t_doall,false,false,_1,_2,_3));
  test("doall N S", M, 16, maxVector, std::bind(t_doall,false,true,_1,_2,_3));
  test("foreach N", M, 16, maxVector, std::bind(t_foreach,false,_1,_2,_3));
  test("doall B W", M, 16, maxVector, std::bind(t_doall,true,false,_1,_2,_3));
  test("doall B S", M, 16, maxVector, std::bind(t_doall,true,true,_1,_2,_3));
  test("foreach B", M, 16, maxVector, std::bind(t_foreach,true,_1,_2,_3));
  return 0;
}
