/** Galois Network Layer for Generalized Buffered Sending -*- C++ -*-
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
#include "Galois/Runtime/NetworkIO.h"

#include <thread>
#include <mutex>
#include <iostream>

using namespace Galois::Runtime;

unsigned _ID = ~0;

namespace {

class NetworkInterfaceBuffered : public NetworkInterface {
  static const int COMM_MIN = 1400; // bytes (sligtly smaller than an ethernet packet)
  static const int COMM_DELAY = 10; //microseconds

  struct recvBuffer {
    std::deque<NetworkIO::message> data;
    size_t frontOffset;
    LL::SimpleLock qlock;
    LL::SimpleLock rlock;

    bool sizeAtLeast(size_t n) {
      size_t tot = -frontOffset;
      for (auto & v : data) {
        tot += v.len;
        if (tot >= n)
          return true;
      }
      return false;
    }

    size_t size() {
      size_t tot = 0;
      for (auto & v : data)
        tot += v.len;
      return tot - frontOffset;
    }

    template<typename IterTy>
    void copyOut(IterTy it, size_t n) {
      assert(sizeAtLeast(n));
      if (n == 0)
        return;
      for (int j = 0; j < data.size(); ++j) {
        auto& v = data[j];
        for (int k = j == 0 ? frontOffset : 0; k < v.len; ++k) {
          *it++ = v.data[k];
          --n;
          if (n == 0)
            return;
        }
      }
      abort();
    }
    
    void erase(size_t n) {
      frontOffset += n;
      while (frontOffset && frontOffset >= data.front().len) {
        frontOffset -= data.front().len;
        data.pop_front();
      }
    }

    uint32_t getLenFromFront() {
      assert(sizeAtLeast(4));
      union { uint8_t a[4]; uint32_t b; } c;
      copyOut(&c.a[0], 4);
      return c.b;
    }

    bool popMsg(std::vector<uint8_t>& vec) {
      std::lock_guard<LL::SimpleLock> lg(qlock);
      if (!sizeAtLeast(4))
        return false;
      uint32_t len = getLenFromFront();
      if (!sizeAtLeast(4 + len))
        return false;
      //std::cerr << _ID << " pm " << frontOffset << " " << size() << " " << len << "\n";
      erase(4);
      vec.reserve(len);
      //FIXME: This is slows things down 25%
      copyOut(std::back_inserter(vec), len);
      erase(len);
      //      std::cerr << _ID << " pp len " << len << " fp " << frontOffset << " sz " << size() << "\n";
      return true;
    }

    //Worker thread interface
    void add(NetworkIO::message m) {
      std::lock_guard<LL::SimpleLock> lg(qlock);
      data.emplace_back(std::move(m));
    }
  };

  std::vector<recvBuffer> recvData;

  struct sendBuffer {
    struct msg {
      uintptr_t fp; // function pointer
      size_t offset; // offset in case ther eis spare space at the front of the buffer
      std::vector<uint8_t> data;
      msg(uintptr_t _fp, size_t _offset, std::vector<uint8_t>& _data) :fp(_fp), offset(_offset) {
        data.swap(_data);
      }
    };

    std::deque<msg> messages;
    std::atomic<size_t> numBytes;
    std::chrono::high_resolution_clock::time_point time;
    std::atomic<bool> urgent;
    LL::SimpleLock lock;

    void markUrgent() {
      urgent = true;
    }

    void add(uintptr_t fp, size_t offset, std::vector<uint8_t>& b) {
      std::lock_guard<LL::SimpleLock> lg(lock);
      if (messages.empty())
        time = std::chrono::high_resolution_clock::now();
      numBytes += b.size() - offset + sizeof(uint32_t) + sizeof(uintptr_t);
      messages.emplace_back(fp, offset, b);
    }

    //Worker thread Interface
    bool ready() {
      if (numBytes == 0)
        return false;
      if (urgent)
        return true;
      if (numBytes > COMM_MIN)
        return true;
      std::lock_guard<LL::SimpleLock> lg(lock);
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time);
      if (elapsed.count() > COMM_DELAY)
        return true;
      return false;
    }
    
    std::pair<std::unique_ptr<uint8_t[]>, size_t> assemble() {
      std::lock_guard<LL::SimpleLock> lg(lock);
      std::pair<std::unique_ptr<uint8_t[]>, size_t> retval;
      retval.second = numBytes;
      retval.first.reset(new uint8_t[numBytes.load()]);
      auto* ii = retval.first.get();
      for (auto& m : messages) {
        uint32_t len = m.data.size() - m.offset + sizeof(uintptr_t);
        for (int i = 0; i < sizeof(uint32_t); ++i)
          *ii++ = ((char*)&len)[i];
        for (int i = 0; i < sizeof(uintptr_t); ++i)
          *ii++ = ((char*)&m.fp)[i];
        for (int i = m.offset; i < m.data.size(); ++i)
          *ii++ = m.data[i];
      }
      messages.clear();
      numBytes = 0;
      urgent = false;
      //      std::cerr << _ID << " as " << retval.second << "\n";
      return retval;
    }
    
  };
    
    std::vector<sendBuffer> sendData;

    void isend(uint32_t dest, uintptr_t fp, SendBuffer& buf) {
      statSendNum += 1;
      statSendBytes += buf.size();
      auto& sd = sendData[dest];
      sd.add(fp, buf.getOffset(), buf.getVec());
    }

  void workerThread() {
    std::unique_ptr<Galois::Runtime::NetworkIO> netio;
    std::tie(netio, ID, Num) = makeNetworkIOMPI();
    _ID = ID;
    ready = 1;
    while (ready < 2) {/*fprintf(stderr, "[WaitOnReady-2]");*/};
    while (ready != 3) {
      for(int i = 0; i < sendData.size(); ++i) {
        netio->progress();
        //handle send queue i
        auto& sd = sendData[i];
        if (sd.ready()) {
          NetworkIO::message msg;
          msg.host = i;
          std::tie(msg.data,msg.len) = sd.assemble();
          netio->enqueue(std::move(msg));
        }
        //handle recieve
        NetworkIO::message rdata = netio->dequeue();
        if (rdata.len)
          recvData[rdata.host].add(std::move(rdata));
      }
    }
  }

  std::thread worker;
  std::atomic<int> ready;

public:
  using NetworkInterface::ID;
  using NetworkInterface::Num;

  NetworkInterfaceBuffered() {
    ready = 0;
    worker = std::thread(&NetworkInterfaceBuffered::workerThread, this);
    while (ready != 1) {/*fprintf(stderr, "[WaitOnReady-1]");*/};
    decltype(sendData) v1(Num);
    decltype(recvData) v2(Num);
    sendData.swap(v1);
    recvData.swap(v2);
    ready = 2;
  }

  virtual ~NetworkInterfaceBuffered() {
    ready = 3;
    worker.join();
  }

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    assert(recv);
    assert(dest < Num);
    //    std::cerr << _ID << " s  " << buf.size() << "\n";
    isend(dest, reinterpret_cast<uintptr_t>(recv), buf);
  }

  virtual void flush() {
    for (auto& sd : sendData)
      sd.markUrgent();
  }

  virtual bool handleReceives() {
    bool retval = false;
    for (auto& rq : recvData) {
      if (rq.rlock.try_lock()) {
        std::lock_guard<LL::SimpleLock> lg(rq.rlock, std::adopt_lock);
        std::vector<uint8_t> data;
        if (rq.popMsg(data)) {
          //          std::cerr << _ID << " hR " << data.size() << "\n";
          retval = true;
          DeSerializeBuffer buf(data);
          statRecvNum += 1;
          statRecvBytes += buf.size();
          uintptr_t fp = 0;
          gDeserialize(buf, fp);
          assert(fp);
          recvFuncTy f = (recvFuncTy)fp;
          f(buf);
        }
      }
    }
    return retval;
  }

};

} //namespace ""

NetworkInterface& Galois::Runtime::makeNetworkBuffered() {
  static NetworkInterfaceBuffered net;
  return net;
}
