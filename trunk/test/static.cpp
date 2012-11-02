// std_tr1__type_traits__is_pod.cpp 

#include <type_traits> 
#include <iostream> 

#include "Galois/Runtime/ll/PtrLock.h"
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/ll/StaticInstance.h"

using namespace GaloisRuntime;
using namespace GaloisRuntime::LL;

int main() 
{ 
  std::cout << "is_pod PtrLock<int, true> == " << std::boolalpha 
	    << std::is_pod<PtrLock<int, true> >::value << std::endl; 
  std::cout << "is_pod PtrLock<int, false> == " << std::boolalpha 
	    << std::is_pod<PtrLock<int, false> >::value << std::endl; 

  std::cout << "is_pod SimpleLock<true> == " << std::boolalpha 
	    << std::is_pod<SimpleLock<true> >::value << std::endl; 
  std::cout << "is_pod SimpleLock<false> == " << std::boolalpha 
	    << std::is_pod<SimpleLock<false> >::value << std::endl; 

  std::cout << "is_pod StaticInstance<int> == " << std::boolalpha 
	    << std::is_pod<StaticInstance<int> >::value << std::endl; 
  std::cout << "is_pod StaticInstance<std::iostream> == " << std::boolalpha 
	    << std::is_pod<StaticInstance<std::iostream> >::value << std::endl; 

  std::cout << "is_pod volatile int == " << std::boolalpha 
   	    << std::is_pod<volatile int>::value << std::endl; 
  std::cout << "is_pod int == " << std::boolalpha 
   	    << std::is_pod<int>::value << std::endl; 

  // std::cout << "is_pod<throws> == " << std::boolalpha 
  // 	    << std::is_pod<throws>::value << std::endl; 
  
  return (0); 
} 
