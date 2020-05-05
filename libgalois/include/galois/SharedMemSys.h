#ifndef GALOIS_SHAREDMEMSYS_H
#define GALOIS_SHAREDMEMSYS_H

#include "galois/config.h"
#include "galois/runtime/SharedMem.h"

namespace galois {

/**
 * SharedMemSys is an explicit class to initialize the Galois runtime. The
 * runtime is destroyed when this object is destroyed.
 */
class SharedMemSys : public runtime::SharedMem<runtime::StatManager> {

public:
  explicit SharedMemSys();
  ~SharedMemSys();

  SharedMemSys(const SharedMemSys&) = delete;
  SharedMemSys& operator=(const SharedMemSys&) = delete;

  SharedMemSys(SharedMemSys&&) = delete;
  SharedMemSys& operator=(SharedMemSys&&) = delete;
};

} // namespace galois

#endif
