/** Galois Distributed Directory -*- C++ -*-
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
 * @author Manoj Dhanapal <madhanap@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/Directory.h"

using namespace Galois::Runtime::Distributed;

uintptr_t RemoteDirectory::haveObject(uintptr_t ptr, uint32_t owner) {
  Lock.lock();
  auto iter = curobj.find(std::make_pair(ptr,owner));
  uintptr_t retval = 0;
  //  if (iter != curobj.end() && iter->state == ObjValid)
  //    retval = iter->localobj;
  Lock.unlock();
  return retval;
}

uintptr_t RemoteDirectory::fetchObject(uintptr_t ptr, uint32_t owner, recvFuncTy pad) {
  abort();
}

RemoteDirectory& getSystemRemoteDirectory() {
  static RemoteDirectory obj;
  return obj;
}

