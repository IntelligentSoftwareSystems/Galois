/** Galois Dedicated Network Thread API -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_NETWORKTHREAD_H
#define GALOIS_RUNTIME_NETWORKTHREAD_H

#include <cstdint>
#include <vector>
#include <tuple>

namespace Galois {
namespace Runtime {

class NetworkIO {
 public:
  virtual ~NetworkIO();
  
  //destructive of data buffer
  virtual void enqueue(uint32_t dest, std::vector<uint8_t>& data) = 0;
  //returns empty if no message
  virtual std::vector<uint8_t> dequeue() = 0;

  //void operator() () -- make progress
  //bool readySend() -- can send
  //bool readyRecv() -- packet waiting
  //void send(const message&) -- send data
  //message recv() -- recieve data
};

std::tuple<NetworkIO*, uint32_t, uint32_t> makeNetworkIOMPI();

} //namespace Runtime
} //namespace Galois

#endif
