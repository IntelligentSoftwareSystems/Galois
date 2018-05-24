#ifndef GALOIS_DIST_GALOIS_H
#define GALOIS_DIST_GALOIS_H

#include "galois/runtime/Init.h"
#include "galois/runtime/DistStats.h"

#include <string>
#include <utility>
#include <tuple>

/**
 * Main Galois namespace. All the core Galois functionality will be found in here.
 */
namespace galois {

/**
 * Explicit class to initialize the Galois Runtime
 * Runtime is destroyed when this object is destroyed
 */
class DistMemSys: public runtime::SharedMemRuntime<runtime::DistStatManager> {

public:
  explicit DistMemSys(void);

  ~DistMemSys(void);
};

} // namespace galois
#endif
