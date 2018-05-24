#include "galois/DistGalois.h"
#include "galois/runtime/Network.h"

galois::DistMemSys::DistMemSys(void)
  : galois::runtime::SharedMemRuntime<galois::runtime::DistStatManager>()
{ }

galois::DistMemSys::~DistMemSys(void) {
  auto& net = galois::runtime::getSystemNetworkInterface();
  net.reportMemUsage();
}
