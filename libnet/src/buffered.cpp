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
#include "Galois/Runtime/Tracer.h"

#include <thread>
#include <mutex>
#include <iostream>

using namespace Galois::Runtime;
using namespace Galois::Substrate;

namespace {

class NetworkInterfaceBuffered : public NetworkInterface {
  static const int COMM_MIN = 1400; // bytes (sligtly smaller than an ethernet packet)
  static const int COMM_DELAY = 100; //microseconds

  unsigned long statSendNum;
  unsigned long statSendBytes;
  unsigned long statSendEnqueued;
  unsigned long statRecvNum;
  unsigned long statRecvBytes;
  unsigned long statRecvDequeued;
  class recvBuffer {
    std::deque<NetworkIO::message> data;
    size_t frontOffset;
    SimpleLock qlock;
    //tag of head of queue
    std::atomic<uint32_t> dataPresent;

    bool sizeAtLeast(size_t n, uint32_t tag) {
      size_t tot = -frontOffset;
      for (auto & v : data) {
        if (v.tag == tag) {
          tot += v.data.size();
          if (tot >= n)
            return true;
        } else {
          return false;
        }
      }
      return false;
    }

    template<typename IterTy>
    void copyOut(IterTy it, size_t n) {
      //assert(sizeAtLeast(n));
      if (n == 0)
        return;
      for (int j = 0; j < data.size(); ++j) {
        auto& v = data[j];
        for (int k = j == 0 ? frontOffset : 0; k < v.data.size(); ++k) {
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
      while (frontOffset && frontOffset >= data.front().data.size()) {
        frontOffset -= data.front().data.size();
        data.pop_front();
      }
      if (data.size()) {
        dataPresent = data.front().tag;
      } else {
        dataPresent = ~0;
      }
    }

    uint32_t getLenFromFront(uint32_t tag) {
      if (sizeAtLeast(sizeof(uint32_t), tag)) {
        union { uint8_t a[sizeof(uint32_t)]; uint32_t b; } c;
        copyOut(&c.a[0], sizeof(uint32_t));
        return c.b;
      } else {
        return ~0;
      }
    }

  public:
    optional_t<RecvBuffer> popMsg(uint32_t tag) {
      std::lock_guard<SimpleLock> lg(qlock);
      uint32_t len = getLenFromFront(tag);
      //      assert(len);
      if (len == ~0 || len == 0)
        return optional_t<RecvBuffer>();
      if (!sizeAtLeast(sizeof(uint32_t) + len, tag))
        return optional_t<RecvBuffer>();
      erase(4);
      RecvBuffer buf(len);
      //FIXME: This is slows things down 25%
      copyOut((char*)buf.linearData(), len);
      erase(len);
      //      std::cerr << "p " << tag << " " << len << "\n";
      return optional_t<RecvBuffer>(std::move(buf));
    }

    //Worker thread interface
    void add(NetworkIO::message m) {
      std::lock_guard<SimpleLock> lg(qlock);
      if (data.empty())
        dataPresent = m.tag;

      //      std::cerr<< m.data.size() << " " << std::count(m.data.begin(), m.data.end(), 0) << "\n";
      // for (auto x : m.data) {
      //   std::cerr << (int) x << " ";
      // }
      // std::cerr << "\n";

      // std::cerr << "A " << m.host << " " << m.tag << " " << m.data.size() << "\n";

      data.push_back(std::move(m));
      assert(data.back().data.size() != std::count(data.back().data.begin(), data.back().data.end(), 0));
    }

    bool hasData(uint32_t tag) {
      return dataPresent == tag;
    }

  };

  std::vector<recvBuffer> recvData;
  std::vector<SimpleLock> recvLock;


  class sendBuffer { 
    struct msg {
      uint32_t tag;
      std::vector<uint8_t> data;
      msg(uint32_t t, std::vector<uint8_t>& _data) :tag(t), data(std::move(_data)) {
      }
    };

    std::deque<msg> messages;
    std::atomic<size_t> numBytes;
    std::atomic<unsigned> urgent;
    std::chrono::high_resolution_clock::time_point time;
    SimpleLock lock;

  public:
    unsigned long statSendTimeout;
    unsigned long statSendOverflow;
    unsigned long statSendUrgent;

    void markUrgent() {
      if (numBytes) {
        std::lock_guard<SimpleLock> lg(lock);
        urgent = messages.size();
      }
    }

    bool ready() {
      if (numBytes == 0)
        return false;
      if (urgent) {
        ++statSendUrgent;
        return true;
      }
      if (numBytes > COMM_MIN) {
        ++statSendOverflow;
        return true;
      }
      std::lock_guard<SimpleLock> lg(lock);
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time);
      if (elapsed.count() > COMM_DELAY) {
        ++statSendTimeout;
        return true;
      }
      return false;      
    }

    std::pair<uint32_t, std::vector<uint8_t> > assemble() {
      std::lock_guard<SimpleLock> lg(lock);
      if (messages.empty())
        return std::make_pair(~0, std::vector<uint8_t>());
      //compute message size
      uint32_t len = 0;
      int num = 0;
      uint32_t tag = messages.front().tag;
      for (auto& m : messages) {
        if (m.tag != tag) {
          break;
        } else {
          len += m.data.size();
          num += sizeof(uint32_t);
        }
      }
      //construct message
      std::vector<uint8_t> vec;
      vec.reserve(len + num);
      while (!messages.empty()) {
        auto& m = messages.front();
        if (m.tag != tag) {
          break;
        } else {
          union {uint32_t a; uint8_t b[sizeof(uint32_t)]; } foo;
          foo.a = m.data.size();
          vec.insert(vec.end(), &foo.b[0], &foo.b[sizeof(uint32_t)]);
          vec.insert(vec.end(), m.data.begin(), m.data.end());
          messages.pop_front();
          if (urgent)
            --urgent;
        }
      }
      numBytes -= len;
      return std::make_pair(tag, std::move(vec));
    }

    void add(uint32_t tag, std::vector<uint8_t>& b) {
      std::lock_guard<SimpleLock> lg(lock);
      if (messages.empty())
        time = std::chrono::high_resolution_clock::now();
      unsigned oldNumBytes = numBytes;
      numBytes += b.size();
      Galois::Runtime::trace("BufferedAdd", oldNumBytes, numBytes, tag, Galois::Runtime::printVec(b));
      messages.emplace_back(tag, b);
    }

  };
    
  std::vector<sendBuffer> sendData;


  
  void workerThread() {
    std::unique_ptr<Galois::Runtime::NetworkIO> netio;
    std::tie(netio, ID, Num) = makeNetworkIOMPI();
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
          std::tie(msg.tag, msg.data) = sd.assemble();
          Galois::Runtime::trace("BufferedSending", msg.host, msg.tag, Galois::Runtime::printVec(msg.data));
          ++statSendEnqueued;
          netio->enqueue(std::move(msg));
        }
        //handle receive
        NetworkIO::message rdata = netio->dequeue();
        if (rdata.data.size()) {
          ++statRecvDequeued;
          assert(rdata.data.size() != std::count(rdata.data.begin(), rdata.data.end(), 0));
          Galois::Runtime::trace("BufferedRecieving", rdata.host, rdata.tag, Galois::Runtime::printVec(rdata.data));
          recvData[rdata.host].add(std::move(rdata));
        }
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
    while (ready != 1) {};
    recvData = decltype(recvData)(Num);
    recvLock.resize(Num);
    sendData = decltype(sendData)(Num);
    ready = 2;
  }

  virtual ~NetworkInterfaceBuffered() {
    ready = 3;
    worker.join();
  }

  virtual void sendTagged(uint32_t dest, uint32_t tag, SendBuffer& buf) {
    statSendNum += 1;
    statSendBytes += buf.size();
    Galois::Runtime::trace("sendTagged", dest, tag, Galois::Runtime::printVec(buf.getVec()));
    auto& sd = sendData[dest];
    sd.add(tag, buf.getVec());
  }

  virtual optional_t<std::pair<uint32_t, RecvBuffer>> recieveTagged(uint32_t tag, std::unique_lock<Galois::Substrate::SimpleLock>* rlg) {
    for (unsigned h = 0; h < recvData.size(); ++h) {
      auto& rq = recvData[h];
      if (rq.hasData(tag)) {
        if (recvLock[h].try_lock()) {
          std::unique_lock<Galois::Substrate::SimpleLock> lg(recvLock[h], std::adopt_lock);
          auto buf = rq.popMsg(tag);
          if (buf) {
            ++statRecvNum;
            statRecvBytes += buf->size();
            if (rlg)
              *rlg = std::move(lg);
            Galois::Runtime::trace("recvTagged", h, tag, Galois::Runtime::printVec(buf->getVec()));
            return optional_t<std::pair<uint32_t, RecvBuffer>>(std::make_pair(h, std::move(*buf)));
          }
        }
      }
    }
    return optional_t<std::pair<uint32_t, RecvBuffer>>();
  }

  virtual void flush() {
    for (auto& sd : sendData)
      sd.markUrgent();
  }

  virtual unsigned long reportSendBytes() const { return statSendBytes; }
  virtual unsigned long reportSendMsgs() const { return statSendNum; }
  virtual unsigned long reportRecvBytes() const { return statRecvBytes; }
  virtual unsigned long reportRecvMsgs() const { return statRecvNum; }
  virtual std::vector<unsigned long> reportExtra() const {
    std::vector<unsigned long> retval(5);
    for (auto& sd : sendData) {
      retval[0] += sd.statSendTimeout;
      retval[1] += sd.statSendOverflow;
      retval[2] += sd.statSendUrgent;
    }
    retval[3] = statSendEnqueued;
    retval[4] = statRecvDequeued;
    return retval;
  }
  virtual std::vector<std::pair<std::string,unsigned long>> reportExtraNamed() const {
    std::vector<std::pair<std::string, unsigned long> > retval(5);
    retval[0].first = "SendTimeout";
    retval[1].first = "SendOverflow";
    retval[2].first = "SendUrgent";
    retval[3].first = "SendEnqueued";
    retval[4].first = "RecvDequeued";
    for (auto& sd : sendData) {
      retval[0].second += sd.statSendTimeout;
      retval[1].second += sd.statSendOverflow;
      retval[2].second += sd.statSendUrgent;
    }
    retval[3].second = statSendEnqueued;
    retval[4].second = statRecvDequeued;
    return retval;
  }

};

} //namespace ""

NetworkInterface& Galois::Runtime::makeNetworkBuffered() {
  static std::atomic<NetworkInterfaceBuffered* > net;
  static Substrate::SimpleLock m_mutex;
  
  auto* tmp = net.load();
  if (tmp == nullptr) {
    std::lock_guard<Substrate::SimpleLock> lock(m_mutex);
    tmp = net.load();
    if (tmp == nullptr) {
      tmp = new NetworkInterfaceBuffered();
      net.store(tmp);
    }
  }
  return *tmp;
}
