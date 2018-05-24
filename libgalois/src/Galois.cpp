#include "galois/Galois.h"

galois::SharedMemSys::SharedMemSys(void)
  : galois::runtime::SharedMemRuntime<galois::runtime::StatManager>()
{ }

galois::SharedMemSys::~SharedMemSys(void) {}
