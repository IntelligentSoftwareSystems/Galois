/** Galois Network Backend Layer -*- C++ -*-
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

#ifndef GALOIS_RUNTIME_NETWORKBACKEND_H
#define GALOIS_RUNTIME_NETWORKBACKEND_H

#include "Galois/Substrate/SimpleLock.h"

#include <cstdint>

#include <boost/intrusive/list.hpp>

namespace Galois {
namespace Runtime {

class NetworkBackend {
public:
  struct SendBlock : public boost::intrusive::list_base_hook<> {
    SendBlock(unsigned char* d) :dest(~0), size(0), data(d) {}
    uint32_t dest, size, tag;
    unsigned char* data;
  };

  typedef boost::intrusive::list<SendBlock> BlockList;

protected:
  uint32_t sz, _ID, _Num;
  Substrate::SimpleLock flLock;
  BlockList freelist;

  NetworkBackend(unsigned size);

public:
  virtual ~NetworkBackend();

  SendBlock* allocSendBlock();
  void freeSendBlock(SendBlock*);

  //! send a block.  data is now owned by the Backend
  virtual void send(SendBlock* data) = 0;
  
  //! recieve a message, data is owned by the caller
  //1 and must be returned to this class
  virtual SendBlock* recv() = 0;

  //! make progress
  virtual void flush(bool block = false) = 0;

  //! returns size used by network
  uint32_t size() const { return sz; }

  uint32_t ID() const { return _ID; }
  uint32_t Num() const { return _Num; }

};


NetworkBackend& getSystemNetworkBackend();

}
}

#endif
