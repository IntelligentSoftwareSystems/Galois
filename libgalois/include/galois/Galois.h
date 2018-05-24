#ifndef GALOIS_GALOIS_H
#define GALOIS_GALOIS_H

#include "galois/Loops.h"
#include "galois/runtime/Init.h"
#include "galois/runtime/Mem.h"


/**
 * Main Galois namespace. All the core Galois functionality will be found in here.
 */
namespace galois {

/**
 * explicit class to initialize the Galois Runtime
 * Runtime is destroyed when this object is destroyed
 */
class SharedMemSys: public runtime::SharedMemRuntime<runtime::StatManager> {

public:
  explicit SharedMemSys(void);

  ~SharedMemSys(void);
};

} // end namespace galois
#endif
