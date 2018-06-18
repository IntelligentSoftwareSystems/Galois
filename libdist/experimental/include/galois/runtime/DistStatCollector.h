/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#if 0 // disabled
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

#endif // GALOIS_RUNTIME_DIST_STAT_COLLECTOR_H
#endif
