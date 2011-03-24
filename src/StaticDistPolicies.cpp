#include "Galois/Runtime/Threads.h"
#include "Galois/Runtime/Support.h"

#include <sched.h>

using namespace GaloisRuntime;

static void genericBindToProcessor(int proc) {
#ifdef __linux__
  cpu_set_t mask;
  /* CPU_ZERO initializes all the bits in the mask to zero. */
  CPU_ZERO( &mask );
  
  /* CPU_SET sets only the bit corresponding to cpu. */
  // void to cancel unused result warning
  (void)CPU_SET( proc, &mask );
  
  /* sched_setaffinity returns 0 in success */
  if( sched_setaffinity( 0, sizeof(mask), &mask ) == -1 )
    reportWarning("Could not set CPU Affinity for thread");
  
  return;
#endif      
  reportWarning("Don't know how to bind thread to cpu on this platform");
}

struct FaradayPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor(int id) {
    //schedule inside each package first
    int carry = 0;
    if (id > 23) {
      id -= 24;
      carry = 24;
    }
    genericBindToProcessor(carry + ((id % 6) * 4) + (id / 6));
  }

  FaradayPolicy() {
    numLevels = 1;
    numThreads = 48;
    numCores = 24;
    levelSize.push_back(4);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 4; ++x)
	for (int i = 0; i < 6; ++i)
	  levelMap.push_back(x);
  }
};

struct VoltaPolicy : public ThreadPolicy {

  virtual void bindThreadToProcessor(int id) {
    //1-1 works on volota
    genericBindToProcessor(id);
  }

  VoltaPolicy() {
    numLevels = 2;
    numThreads = 64;
    numCores = 32;

    //NUMA Nodes
    levelSize.push_back(4);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 4; ++x)
	for (int i = 0; i < 8; ++i)
	  levelMap.push_back(x);

    //Packages
    levelSize.push_back(8);
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 8; ++x)
	for (int i = 0; i < 4; ++i)
	  levelMap.push_back(x);
  }
};

static FaradayPolicy a_FaradayPolicy;
static VoltaPolicy a_VoltaPolicy;


ThreadPolicy& GaloisRuntime::getSystemThreadPolicy() {
  return a_FaradayPolicy;
}
