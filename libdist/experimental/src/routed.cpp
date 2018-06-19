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

#include "galois/runtime/Network.h"
#include "galois/runtime/NetworkIO.h"

#include <thread>
#include <mutex>

#if 0

using namespace galois::runtime;

namespace {

class NetworkInterfaceRouted : public NetworkInterface {
  static const int COMM_MIN = 1400; // bytes (sligtly smaller than an ethernet packet)
  static const int COMM_DELAY = 100; //microseconds

  std::unique_ptr<galois::runtime::NetworkIO> netio;

  struct recvBuffer {
    std::deque<uint8_t> data;
    LL::SimpleLock lock;
    
    std::pair<std::deque<uint8_t>::iterator, std::deque<uint8_t>::iterator>
    nextMsg() {
      std::lock_guard<LL::SimpleLock> lg(lock);
      if (data.empty())
        return std::make_pair(data.end(), data.end());
      assert(data.size() >= 8);
      union { uint8_t a[4]; uint32_t b; } c;
      std::copy_n(data.begin(), 4, &c.a[0]);
      return std::make_pair(data.begin() + 4, data.begin() + 4 + c.b);
    }

    void popMsg() {
      std::lock_guard<LL::SimpleLock> lg(lock);
      assert(data.size() >= 8);
      union { uint8_t a[4]; uint32_t b; } c;
      std::copy_n(data.begin(), 4, &c.a[0]);
      data.erase(data.begin(), data.begin() + 4 + c.b);
    }

    //Worker thread interface
    void add(std::vector<uint8_t>& buf) {
      std::lock_guard<LL::SimpleLock> lg(lock);
      data.insert(data.end(), buf.begin(), buf.end());
    }
  };

  recvBuffer recvData;
  LL::SimpleLock recvLock;

  struct sendBuffer {
    std::vector<uint8_t> data;
    std::chrono::high_resolution_clock::time_point time;
    std::atomic<bool> urgent;
    LL::SimpleLock lock;

    void markUrgent() {
      urgent = true;
    }

    void add(SendBuffer& b) {
      std::lock_guard<LL::SimpleLock> lg(lock);
      if (data.empty())
        time = std::chrono::high_resolution_clock::now();
      union { uint8_t a[4]; uint32_t b; } c;
      c.b = b.size();
      data.insert(data.end(), &c.a[0], &c.a[4]);
      data.insert(data.end(), (uint8_t*)b.linearData(), (uint8_t*)b.linearData() + b.size());
    }

    //Worker thread Interface
    bool ready() {
      std::lock_guard<LL::SimpleLock> lg(lock);
      if (data.empty())
        return false;
      if (urgent)
        return true;
      if (data.size() > COMM_MIN)
        return true;
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time);
      if (elapsed.count() > COMM_DELAY)
        return true;
      return false;
    }

  };

  std::vector<sendBuffer> sendData;

  uint32_t getRouter(uint32_t dest) const {
    if (isRouter())
      return dest;
    else
      return Num;
  }

  bool isRouter() const {
    return ID == Num;
  }

  void isend(uint32_t dest, SendBuffer& buf) {
    statSendNum += 1;
    statSendBytes += buf.size();
    uint32_t rdest = getRouter(dest);
    auto& sd = sendData[rdest];
    sd.add(buf);
  }

  void workerThread() {
    std::tie(netio, ID, Num) = makeNetworkIOMPI();
    ready = 1;
    while (ready != 2) {};
    while (ready != 3) {
      do {
        std::vector<uint8_t> rdata = netio->dequeue();
        if (rdata.empty())
          break;
        else
          recvData.add(rdata);
      } while (true);
      for(int i = 0; i < sendData.size(); ++i) {
        auto& sd = sendData[i];
        if (sd.ready()) {
          std::lock_guard<LL::SimpleLock> lg(sd.lock);
          netio->enqueue(i, sd.data);
          assert(sd.data.empty());
          sd.urgent = false;
        }
      }
    }
  }

  std::thread worker;
  std::atomic<int> ready;

public:
  using NetworkInterface::ID;
  using NetworkInterface::Num;

  NetworkInterfaceRouted() {
    ready = 0;
    worker = std::thread(&NetworkInterfaceRouted::workerThread, this);
    while (ready != 1) {};
    decltype(sendData) v(Num);
    sendData.swap(v);
    --Num;
    ready = 2;
  }

  virtual ~NetworkInterfaceRouted() {
    ready = 3;
    worker.join();
  }

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    assert(recv);
    assert(dest < Num);
    buf.serialize_header((void*)recv);
    uintptr_t pdest = dest;
    buf.serialize_header((void*)pdest);
    isend(dest, buf);
  }

  virtual void flush() {
    if(!isRouter())
      sendData[getRouter(ID)].markUrgent();
  }

  virtual bool handleReceives() {
    bool retval = false;
    if (recvLock.try_lock()) {
      std::lock_guard<LL::SimpleLock> lg(recvLock, std::adopt_lock);
      auto p = recvData.nextMsg();
      if (std::get<0>(p) != std::get<1>(p)) {
        retval = true;
        DeSerializeBuffer buf(std::get<0>(p), std::get<1>(p));
        statRecvNum += 1;
        statRecvBytes += buf.size();
        recvData.popMsg();
        uintptr_t dest;
        uintptr_t fp = 0;
        gDeserialize(buf, dest, fp);
        assert(fp);
        recvFuncTy f = (recvFuncTy)fp;
        if (dest == ID) { //deliver for us
          f(buf);
        } else { //route for others
          SerializeBuffer buf2{std::move(buf)};
          send(dest, f, buf2);
        }
      }
    }
    return retval;
  }
};

} //namespace ""

NetworkInterface& galois::runtime::makeNetworkRouted() {
  static NetworkInterfaceRouted net;
  return net;
}

#endif
