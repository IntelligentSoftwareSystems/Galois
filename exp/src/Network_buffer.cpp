/** Galois Network Layer for Message Aggregation -*- C++ -*-
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

#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/NetworkBackend.h"
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/gstl.h"

using namespace Galois;
using namespace Galois::Runtime;

namespace {

class NetworkInterfaceBuffer : public NetworkInterface {

  NetworkBackend& net;

  struct header {
    uint32_t size;
    uintptr_t func;
  };

  struct state {
    LL::SimpleLock<true> lock_out;
    LL::SimpleLock<true> lock_in;
    NetworkBackend::SendBlock* cur_out;
    NetworkBackend::SendBlock* cur_in;
    state() :cur_out(nullptr), cur_in(nullptr) {}
  };

  std::vector<state> states;
  LL::SimpleLock<true> orderingLock;

  //returns number of unwritten bytes
  unsigned writeInternal(NetworkBackend::SendBlock* buf, char* data, unsigned len) {
    auto eiter = safe_copy_n(data, data + len, net.size() - buf->size, buf->data + buf->size);
    unsigned copyLen = eiter - (buf->data + buf->size);
    buf->size += copyLen;
    return len - copyLen;
  }
    
  void writeBuffer(state& s, char* data, unsigned len) {
    if (!s.cur_out)
      s.cur_out = net.allocSendBlock();
    do {
      unsigned c = writeInternal(s.cur_out, data, len);
      data += c;
      len -= c;
      if (c) {
        net.send(s.cur_out);
        s.cur_out = net.allocSendBlock();
      }
    } while (len);
  }

  void sendInternal(state& s, recvFuncTy recv, SendBuffer& buf) {
    std::lock_guard<LL::SimpleLock<true>> lg(s.lock_out);
    header h = {(uint32_t)buf.size(), (uintptr_t)recv};
    writeBuffer(s, (char*)&h, sizeof(header));
    writeBuffer(s, (char*)buf.linearData(), buf.size());
  }

public:

  NetworkInterfaceBuffer()
    :net(getSystemNetworkBackend())
  {
    states.resize(net.Num());
    ID = net.ID();
    Num = net.Num();
  }

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    sendInternal(states[dest], recv, buf);
  }

  virtual bool handleReceives() {
    orderingLock.lock();
    //empty network
    NetworkBackend::SendBlock* b = net.recv();
    while (b) {
      //append
      NetworkBackend::SendBlock** head = &states[b->dest].cur_in;
      while (*head) head = &((*head)->next);
      *head = b;
      b = net.recv();
    }
    orderingLock.unlock();

    //deliver messages

    return false;
  }

};

}

#ifdef USE_BUF
NetworkInterface& Galois::Runtime::getSystemNetworkInterface() {
  static NetworkInterfaceBuffer net;
  return net;
}
#endif
