/** Simple pool to manage memory for contexts -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_CONTEXTPOOL_H
#define GALOIS_RUNTIME_CONTEXTPOOL_H

#include "Galois/Runtime/WorkList.h"

namespace GaloisRuntime {

template<typename Context>
class ContextPool {
  typedef WorkList::dChunkedLIFO<1024,Context> Pool;
  Pool pool;
public:
  ~ContextPool() {
    boost::optional<Context> p;
    while ((p = pool.pop()))
      ;
  }

  Context* next() {
    Context* retval = pool.push(Context());
    return retval;
  }

  void commitAll() {
    boost::optional<Context> p;
    while ((p = pool.pop())) {
      p->commit_iteration();
    }
  }
};

}
#endif
