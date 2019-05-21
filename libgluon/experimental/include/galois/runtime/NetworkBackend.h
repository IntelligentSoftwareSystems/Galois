/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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

#ifndef GALOIS_RUNTIME_NETWORKBACKEND_H
#define GALOIS_RUNTIME_NETWORKBACKEND_H

#include "galois/substrate/SimpleLock.h"
#include <cstdint>
#include <boost/intrusive/list.hpp>

namespace galois {
namespace runtime {

class NetworkBackend {
public:
  struct SendBlock : public boost::intrusive::list_base_hook<> {
    SendBlock(unsigned char* d) : dest(~0), size(0), data(d) {}
    uint32_t dest, size, tag;
    unsigned char* data;
  };

  typedef boost::intrusive::list<SendBlock> BlockList;

protected:
  uint32_t sz, _ID, _Num;
  substrate::SimpleLock flLock;
  BlockList freelist;

  NetworkBackend(unsigned size);

public:
  virtual ~NetworkBackend();

  SendBlock* allocSendBlock();
  void freeSendBlock(SendBlock*);

  //! Send a block; data is now owned by the Backend
  virtual void send(SendBlock* data) = 0;

  //! Recieve a message; data is now owned by the caller
  //! and must be returned to this class
  virtual SendBlock* recv() = 0;

  //! Make progress
  virtual void flush(bool block = false) = 0;

  //! returns size used by network
  uint32_t size() const { return sz; }

  uint32_t ID() const { return _ID; }
  uint32_t Num() const { return _Num; }
};

NetworkBackend& getSystemNetworkBackend();

// implementations copied over from Network.cpp
// NetworkBackend::SendBlock* NetworkBackend::allocSendBlock() {
//  //FIXME: review for TBAA rules
//  std::lock_guard<substrate::SimpleLock> lg(flLock);
//  SendBlock* retval = nullptr;
//  if (freelist.empty()) {
//    unsigned char* data = (unsigned char*)malloc(sizeof(SendBlock) + size());
//    retval = new (data) SendBlock(data + sizeof(SendBlock));
//  } else {
//    retval = &freelist.front();
//    freelist.pop_front();
//    retval->size = 0;
//    retval->dest = ~0;
//  }
//  return retval;
//}
//
// void NetworkBackend::freeSendBlock(SendBlock* sb) {
//  std::lock_guard<substrate::SimpleLock> lg(flLock);
//  freelist.push_front(*sb);
//}
//
// NetworkBackend::~NetworkBackend() {
//  while (!freelist.empty()) {
//    SendBlock* sb = &freelist.front();
//    freelist.pop_front();
//    sb->~SendBlock();
//    free(sb);
//  }
//}
//
// NetworkBackend::NetworkBackend(unsigned size) :sz(size),_ID(0),_Num(0) {}

} // namespace runtime
} // namespace galois
#endif
