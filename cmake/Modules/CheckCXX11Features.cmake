include(CheckCXXSourceCompiles)
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

set(CMAKE_REQUIRED_FLAGS ${CXX11_FLAGS})
CHECK_CXX_SOURCE_COMPILES("${CheckUniformIntDistribution}"
  HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
CHECK_CXX_SOURCE_COMPILES("${CheckUniformRealDistribution}"
  HAVE_CXX11_UNIFORM_REAL_DISTRIBUTION)
