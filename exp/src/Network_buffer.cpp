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

using namespace Galois;
using namespace Galois::Runtime;

namespace {

class NetworkInterfaceBuffer : public NetworkInterface {

  NetworkBackend& net;

  struct header {
    uint32_t size;
    uintptr_t func;
  };

  struct state_in {
    LL::SimpleLock<true> lock;
    NetworkBackend::SendBlock* cur;
    unsigned offset;
    state_in() :cur(nullptr), offset(0) {}
  };
  struct state_out {
    LL::SimpleLock<true> lock;
    NetworkBackend::SendBlock* cur;
    state_out() :cur(nullptr) {}
  };
  struct state {
    state_in in;
    state_out out;
  };

  std::vector<state> states;
  //! preserves recieve ordering from the network and ensures serialized access
  LL::SimpleLock<true> orderingLock;

  void writeBuffer(state_out& s, char* data, unsigned len) {
    if (!s.cur)
      s.cur = net.allocSendBlock();
    auto ptr = s.cur;
    while (len) {
      while (ptr->size == net.size()) {
        if (!ptr->next)
          ptr->next = net.allocSendBlock();
        ptr = ptr->next;
      }
      unsigned toCopy = std::min(len, net.size() - ptr->size);
      std::copy_n(data, toCopy, ptr->data + ptr->size);
      data += toCopy;
      len -= toCopy;
      ptr->size += toCopy;
    }
  }

  void sendInternal(state_out& s, unsigned dest, recvFuncTy recv, SendBuffer& buf) {
    std::lock_guard<LL::SimpleLock<true>> lg(s.lock);
    header h = {(uint32_t)buf.size(), (uintptr_t)recv};
    writeBuffer(s, (char*)&h, sizeof(header));
    writeBuffer(s, (char*)buf.linearData(), buf.size());
    //write out to network
    while (s.cur && s.cur->size == net.size()) {
      auto ptr = s.cur;
      s.cur = ptr->next;
      ptr->dest = dest;
      net.send(ptr);
    }
  }


  //non advancing read form offset of len into data
  void readBuffer(state_in& s, unsigned offset, char* data, unsigned len) {
    auto ptr = s.cur;
    unsigned off = offset + s.offset;
    while (len) {
      while (off >= ptr->size) { // may be in next block
        off -= ptr->size;
        ptr = ptr->next;
      }
      unsigned toCopy = std::min(len, ptr->size - off);
      std::copy_n(ptr->data + off, toCopy, data);
      data += toCopy;
      off += toCopy;
      len -= toCopy;
    }
  }

  unsigned readLen(state_in& s) {
    unsigned sum = 0;
    NetworkBackend::SendBlock* h = s.cur;
    while (h) {
      sum += h->size;
      h = h->next;
    }
    return sum - s.offset;
  }

  void advBuffer(state_in& s, unsigned len) {
    s.offset += len;
    while (s.offset && s.offset >= s.cur->size) {
      auto old = s.cur;
      s.cur = old->next;
      s.offset -= old->size;
      net.freeSendBlock(old);
    }
 }

  void recvInternal(state_in& s) {
    if (s.lock.try_lock()) {
      std::lock_guard<LL::SimpleLock<true>> lg(s.lock, std::adopt_lock);
      header h;
      if (readLen(s) < sizeof(header)) return;
      readBuffer(s, 0, (char*)&h, sizeof(header));
      if (readLen(s) < sizeof(header) + h.size) return;
      RecvBuffer buf(h.size);
      readBuffer(s, sizeof(header), (char*)buf.linearData(), h.size);
      advBuffer(s, sizeof(header) + h.size);
      recvFuncTy func = (recvFuncTy)h.func;
      func(buf);
    }
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
    sendInternal(states[dest].out, dest, recv, buf);
  }

  virtual bool handleReceives() {
    if (orderingLock.try_lock()) {
      std::lock_guard<LL::SimpleLock<true>> lg(orderingLock, std::adopt_lock);
      //empty network
      NetworkBackend::SendBlock* b = nullptr;
      while ((b = net.recv())) {
        //append
        std::lock_guard<LL::SimpleLock<true>> lg(states[b->dest].in.lock);
        NetworkBackend::SendBlock** head = &states[b->dest].in.cur;
        while (*head) head = &((*head)->next);
        *head = b;
      }
    }

    //deliver messages

    for (int i = 0; i < states.size(); ++i)
      recvInternal(states[i].in);

    return false;
  }

  virtual void flush() {
    for (int i = 0; i < states.size(); ++i) {
      state_out& s = states[i].out;
      std::lock_guard<LL::SimpleLock<true>> lg(s.lock);
      if (s.cur && s.cur->size) {
        s.cur->dest = i;
        net.send(s.cur);
        s.cur = net.allocSendBlock();
      }
    }
  }

};

}

#ifdef USE_BUF
NetworkInterface& Galois::Runtime::getSystemNetworkInterface() {
  static NetworkInterfaceBuffer net;
  return net;
}
#endif
