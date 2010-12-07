// -*- C++ -*-
#include <linux/mman.h>
#include <sys/mman.h>


namespace GaloisRuntime {
namespace MM {

class mmapWrapper {
  static void* alloc();
  static void free();
public:
  enum {AllocSize = 2*1024*1024,
	Alighment = 4*1024};
  
  void* allocate(unsigned int size);
  void deallocate(void* ptr);
};

}
}
