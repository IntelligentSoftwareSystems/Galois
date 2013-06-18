include(CheckCXXSourceCompiles)
set(CheckUniformDistribution
"
#include <random>
int main(){
  std::mt19937 gen;
  std::uniform_int_distribution<int> r(0, 6);
  return r(gen);
}
")

set(CMAKE_REQUIRED_FLAGS ${CXX11_FLAGS})
CHECK_CXX_SOURCE_COMPILES("${CheckUniformDistribution}"
  HAVE_CXX11_UNIFORM_INT_DISTRIBUTION)
