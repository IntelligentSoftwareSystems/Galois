/** Galois Network Layer -*- C++ -*-
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

#ifndef GALOIS_RUNTIME_NETWORK_H
#define GALOIS_RUNTIME_NETWORK_H

#include "Galois/Runtime/Serialize.h"

#include <cstdint>

namespace Galois {
namespace Runtime {
namespace Distributed {

extern uint32_t networkHostID;

typedef SerializeBuffer SendBuffer;
class RecvBuffer {};

typedef void (*recvFuncTy)(RecvBuffer&);

class NetworkInterface {
public:
  virtual ~NetworkInterface() {}

  //!send a message to a given (dest) host.  A message is simply a
  //!landing pad (recv) and some data (buf)
  //! buf is invalidated by this operation
  virtual void sendMessage(uint32_t dest, recvFuncTy recv, SendBuffer& buf) = 0;

  //!send a message to all hostss.  A message is simply a
  //!landing pad (recv) and some data (buf)
  //! buf is invalidated by this operation
  virtual void broadcastMessage(recvFuncTy recv, SendBuffer& buf) = 0;
};

NetworkInterface& getSystemNetworkInterface();


} //Distributed
} //Runtime
} //Galois
#endif
