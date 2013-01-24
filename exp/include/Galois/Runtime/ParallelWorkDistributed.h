/** Galois Distributed Loop -*- C++ -*-
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
 * Implementation of the Galois distributed foreach iterator.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_PARALLELWORKDIST_H
#define GALOIS_RUNTIME_PARALLELWORKDIST_H

#include "Galois/Runtime/Network.h"

namespace Galois {
namespace Runtime {

template<typename WLTy, typename ItemTy, typename FunctionTy>
void for_each_landing_pad(Distributed::RecvBuffer& buf) {
  //extract stuff
  FunctionTy f;
  buf.deserialize(f);
  std::deque<ItemTy> data;
  buf.deserialize(data);
printf ("Host %u, number of elements passed %lu\n", Distributed::networkHostID, data.size());
  //Start locally
  Galois::Runtime::for_each_impl<WLTy>(data.begin(), data.end(), f, nullptr);
}

namespace {

template<typename WLTy, typename IterTy, typename FunctionTy>
void for_each_dist(IterTy b, IterTy e, FunctionTy f, const char* loopname) {
  //fast path for non-distributed
  if (Distributed::networkHostNum == 1) {
    for_each_impl<WLTy>(b,e,f,loopname);
    return;
  }


  typedef typename std::iterator_traits<IterTy>::value_type ItemTy;

  //copy out all data
  std::deque<ItemTy> allData(b,e);

  // Get a handle to the network interface
  Distributed::NetworkInterface& net = Distributed::getSystemNetworkInterface();
  for (unsigned i = 1; i < Distributed::networkHostNum; i++) {
    auto blk = block_range(allData.begin(), allData.end(), i, Distributed::networkHostNum);
    std::deque<ItemTy> data(blk.first, blk.second);
    Distributed::SendBuffer buf;
    // serialize function
    buf.serialize(f);
    // serialize data
    buf.serialize(data);
    //send data
    net.sendMessage (i, &for_each_landing_pad<WLTy,ItemTy,FunctionTy>, buf);
    // send the data
    net.handleReceives();
  }
  //now get our data
  auto myblk = block_range(allData.begin(), allData.end(), 0, Distributed::networkHostNum);

  //Start locally
  for_each_impl<WLTy>(myblk.first, myblk.second, f, loopname);
}


} // anon
} // Runtime
} // Galois

#endif
