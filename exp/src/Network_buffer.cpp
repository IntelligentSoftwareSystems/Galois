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
#include "Galois/Runtime/mm/Mem.h"
#include "Galois/gstl.h"

using namespace Galois;
using namespace Galois::Runtime;

namespace {

class NetworkInterfaceBuffer : public NetworkInterface {

  typedef std::pair<char*, unsigned> bufEntry;
  
  struct header {
    uint32_t size;
    uintptr_t func;
  };

  struct state {
    Galois::Runtime::LL::SimpleLock<true> lock;
    std::deque<bufEntry> buffers;
  };

  std::vector<state> states;
  MM::FixedSizeAllocator alloc;

  unsigned sizeBuffer() const;
  char* allocBuffer() { return (char*)alloc.allocate(sizeBuffer()); }
  void freeBuffer(char* b) { alloc.deallocate(b); }

  //returns number of unwritten bytes
  unsigned writeInternal(bufEntry& buf, char* data, unsigned len) {
    auto eiter = safe_copy_n(data, data + len, buf.second,
                             buf.first + (sizeBuffer() - buf.second));
    unsigned copyLen = eiter - data;
    buf.second -= copyLen;
    return len - copyLen;
  }
    
  void writeBuffer(state& s, char* data, unsigned len) {
    if (s.buffers.empty())
      s.buffers.push_back(std::make_pair(allocBuffer(), sizeBuffer()));
    do {
      bufEntry& b = s.buffers.back();
      unsigned c = writeInternal(b, data, len);
      data += c;
      len -= c;
      if (c) {
        s.buffers.push_back(std::make_pair(allocBuffer(), sizeBuffer()));
      }
    } while (len);
  }

  void sendInternal(state& s, recvFuncTy recv, SendBuffer& buf) {
    std::lock_guard<LL::SimpleLock<true>> lg(s.lock);
    header h = {(uint32_t)buf.size(), (uintptr_t)recv};
    writeBuffer(s, (char*)&h, sizeof(header));
    writeBuffer(s, (char*)buf.linearData(), buf.size());
  }

public:

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    sendInternal(states[dest], recv, buf);
  }

  virtual bool handleReceives() = 0;

};

}

