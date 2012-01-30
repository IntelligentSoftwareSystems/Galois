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

#ifndef _TID_H
#define _TID_H

namespace GaloisRuntime {
namespace LL {

  extern __thread unsigned TID;
  extern unsigned nextID;

  //Get this thread's id.
  static unsigned getTID() {
    unsigned x = TID;
    if (x & 1)
      return x >> 1;
    x = __sync_fetch_and_add(&nextID, 1);
    TID = (x << 1) | 1;
    return x;
  }
}
}

#endif //_TID_H
