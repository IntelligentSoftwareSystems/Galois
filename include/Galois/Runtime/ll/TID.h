/** Thread ID -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights
 * reserved.  UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
 * CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE,
 * NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY WARRANTY
 * THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF
 * TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO
 * THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect,
 * direct or consequential damages or loss of profits, interruption of
 * business, or related expenses which may arise from use of Software
 * or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of
 * data of any kind.  
 *
 * @section Description
 *
 * Manage Thread ID.  ID's are sequential and dense from zero.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_LL_TID_H
#define GALOIS_RUNTIME_LL_TID_H

namespace Galois {
namespace Runtime {
//! Low-level, Galois-agnostic functionality.
namespace LL {

extern __thread unsigned TID;

//Get this thread's id.
static inline unsigned getTID() {
  return TID;
}

//uninitialized TIDs are 0, and only thread 0 accesses TID before
//initializing it
void initTID();

#ifdef GALOIS_USE_EXP
void initTID_cilk();
#endif // GALOIS_USE_EXP

}
}
}

#endif //_TID_H
