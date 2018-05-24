#if 0// disabled
#ifndef GALOIS_RUNTIME_DIST_STAT_COLLECTOR_H
#define GALOIS_RUNTIME_DIST_STAT_COLLECTOR_H

#include "galois/runtime/StatCollector.h"
#include "galois/runtime/Substrate.h"
#include "galois/runtime/Serialize.h"
#include "galois/runtime/Network.h"

namespace galois {
namespace runtime {


class DistStatCollector: public StatCollector {
protected:

  using Base = StatCollector;

  using Base::RecordTy;

public:

  void printStats(void);

  DistStatCollector(const std::string& outfile="");

private:

  void combineAtHost_0(void);

};

} // end namespace runtime
} // end namespace galois

#endif// GALOIS_RUNTIME_DIST_STAT_COLLECTOR_H
#endif
