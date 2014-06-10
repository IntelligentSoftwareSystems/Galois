include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

set(CheckUniformIntDistribution
"
#include <random>
int main(){
  std::mt19937 gen;
  std::uniform_int_distribution<int> r(0, 6);
  return r(gen);
}
")

set(CheckUniformRealDistribution
"
#include <random>
int main(){
  std::mt19937 gen;
  std::uniform_real_distribution<float> r(0, 1);
  return r(gen) < 0.5 ? 0 : 1;
}
")

set(CheckChrono
"
#include <chrono>
int main(){
  typedef  std::chrono::steady_clock Clock;
  std::chrono::time_point<Clock> start, stop;
  start = Clock::now();
  stop = Clock::now();
  unsigned long res = 
     std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
  return res < 1000 ? 0 : 1;
}
")

set(CheckAlignof
"
int main(){
  return alignof(int) != 0;
}
")

set(CheckThreadYield
"
#include <thread>
int main() {
  std::this_thread::yield();
  return 0;
}
")

cmake_push_check_state()

set(CMAKE_REQUIRED_FLAGS ${CXX11_FLAGS})
CHECK_CXX_SOURCE_COMPILES("${CheckUniformIntDistribution}"
  HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
CHECK_CXX_SOURCE_COMPILES("${CheckUniformRealDistribution}"
  HAVE_CXX11_UNIFORM_REAL_DISTRIBUTION)
CHECK_CXX_SOURCE_COMPILES("${CheckChrono}"
  HAVE_CXX11_CHRONO)
CHECK_CXX_SOURCE_COMPILES("${CheckAlignof}"
  HAVE_CXX11_ALIGNOF)
CHECK_CXX_SOURCE_COMPILES("${CheckThreadYield}"
  HAVE_CXX11_THREAD_YIELD)

cmake_pop_check_state()
