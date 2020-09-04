/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

/*

 @Vinicius Possani
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#ifndef RECONVDRIVENCUT_H_
#define RECONVDRIVENCUT_H_

#include "Aig.h"

#include <unordered_set>

namespace algorithm {

typedef struct RDCutData_ {

  std::unordered_set<aig::GNode> visited;
  std::unordered_set<aig::GNode> leaves;

} RDCutData;

typedef galois::substrate::PerThreadStorage<RDCutData> PerThreadRDCutData;

class ReconvDrivenCut {

private:
  aig::Aig& aig;
  PerThreadRDCutData perThreadRDCutData;

public:
  ReconvDrivenCut(aig::Aig& aig);

  virtual ~ReconvDrivenCut();

  void run(size_t cutSizeLimit);
};

} /* namespace algorithm */

namespace alg = algorithm;

#endif /* RECONVDRIVENCUT_H_ */
