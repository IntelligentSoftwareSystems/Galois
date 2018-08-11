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

/**
 * @file NetworkBuffered.cpp
 *
 * Contains NetworkInterfaceBuffered, an implementation of a network interface
 * that buffers messages before sending them out.
 *
 * @todo document this file more
 */

#include "galois/runtime/Network.h"
#include "galois/runtime/NetworkIO.h"
#include "galois/runtime/Tracer.h"

#ifdef GALOIS_USE_LWCI
#define NO_AGG
#endif

#include <thread>
#include <mutex>
#include <iostream>
#include <limits>

using namespace galois::runtime;
using namespace galois::substrate;

namespace {

/**
 * @class NetworkInterfaceBuffered
 *
 * Buffered network interface: messages are buffered before they are sent out.
 * A single worker thread is initialized to send/receive messages from/to
 * buffers.
 */
class NetworkInterfaceBuffered : public NetworkInterface {
  static const int COMM_MIN =
      1400; //! bytes (sligtly smaller than an ethernet packet)
  static const int COMM_DELAY = 100; //! microseconds delay

  unsigned long statSendNum;
  unsigned long statSendBytes;
  unsigned long statSendEnqueued;
  unsigned long statRecvNum;
  unsigned long statRecvBytes;
  unsigned long statRecvDequeued;
  bool anyReceivedMessages;

  /**
   * Receive buffers for the buffered network interface
   */
  class recvBuffer {
    std::deque<NetworkIO::message> data;
    size_t frontOffset;
    SimpleLock qlock;
    // tag of head of queue
    std::atomic<uint32_t> dataPresent;

    bool sizeAtLeast(size_t n, uint32_t tag) {
      size_t tot = -frontOffset;
      for (auto& v : data) {
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

    template <typename IterTy>
    void copyOut(IterTy it, size_t n) {
      // assert(sizeAtLeast(n));
      // fast path is first buffer
      { // limit scope
        auto& f0data = data[0].data;
        for (int k = frontOffset, ke = f0data.size(); k < ke && n; ++k, --n)
          *it++ = f0data[k];
      }
      if (n) { // more data (slow path)
        for (int j = 1, je = data.size(); j < je && n; ++j) {
          auto& vdata = data[j].data;
          for (int k = 0, ke = vdata.size(); k < ke && n; ++k, --n) {
            *it++ = vdata[k];
          }
        }
      }
    }

    /**
     * Return a (moved) vector if the len bytes requested are the last len
     * bytes of the front of the buffer queue
     */
    optional_t<std::vector<uint8_t>> popVec(uint32_t len) {
      if (data[0].data.size() == frontOffset + len) {
        std::vector<uint8_t> retval(std::move(data[0].data));
        data.pop_front();
        frontOffset = 0;
        if (data.size()) {
          dataPresent = data.front().tag;
        } else {
          dataPresent = ~0;
        }
        return optional_t<std::vector<uint8_t>>(std::move(retval));
      } else {
        return optional_t<std::vector<uint8_t>>();
      }
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
        union {
          uint8_t a[sizeof(uint32_t)];
          uint32_t b;
        } c;
        copyOut(&c.a[0], sizeof(uint32_t));
        return c.b;
      } else {
        return ~0;
      }
    }

  public:
    optional_t<RecvBuffer> popMsg(uint32_t tag) {
      std::lock_guard<SimpleLock> lg(qlock);
#ifndef NO_AGG
      uint32_t len = getLenFromFront(tag);
      //      assert(len);
      if (len == ~0U || len == 0)
        return optional_t<RecvBuffer>();
      if (!sizeAtLeast(sizeof(uint32_t) + len, tag))
        return optional_t<RecvBuffer>();
      erase(4);

      // Try just using the buffer
      if (auto r = popVec(len)) {
        auto start = r->size() - len;
        //        std::cerr << "FP " << r->size() << " " << len << " " << start
        //        << "\n";
        return optional_t<RecvBuffer>(RecvBuffer(std::move(*r), start));
      }

      RecvBuffer buf(len);
      // FIXME: This is slows things down 25%
      copyOut((char*)buf.linearData(), len);
      erase(len);
      // std::cerr << "p " << tag << " " << len << "\n";
      return optional_t<RecvBuffer>(std::move(buf));
#else
      if (data.empty() || data.front().tag != tag)
        return optional_t<RecvBuffer>();

      std::vector<uint8_t> vec(std::move(data.front().data));

      data.pop_front();
      if (!data.empty()) {
        dataPresent = data.front().tag;
      } else {
        dataPresent = ~0;
      }

      return optional_t<RecvBuffer>(RecvBuffer(std::move(vec), 0));
#endif
    }

    // Worker thread interface
    void add(NetworkIO::message m) {
      std::lock_guard<SimpleLock> lg(qlock);
      if (data.empty()) {
        galois::runtime::trace("ADD LATEST ", m.tag);
        dataPresent = m.tag;
      }

      // std::cerr << m.data.size() << " " <<
      //              std::count(m.data.begin(), m.data.end(), 0) << "\n";
      // for (auto x : m.data) {
      //   std::cerr << (int) x << " ";
      // }
      // std::cerr << "\n";
      // std::cerr << "A " << m.host << " " << m.tag << " " << m.data.size() <<
      // "\n";

      data.push_back(std::move(m));

      assert(data.back().data.size() !=
             (unsigned int)std::count(data.back().data.begin(),
                                      data.back().data.end(), 0));
    }

    bool hasData(uint32_t tag) { return dataPresent == tag; }

    size_t size() { return data.size(); }

    uint32_t getPresentTag() { return dataPresent; }
  }; // end recv buffer class

  std::vector<recvBuffer> recvData;
  std::vector<SimpleLock> recvLock;

  /**
   * Send buffers for the buffered network interface
   */
  class sendBuffer {
    struct msg {
      uint32_t tag;
      std::vector<uint8_t> data;
      msg(uint32_t t, std::vector<uint8_t>& _data)
          : tag(t), data(std::move(_data)) {}
    };

    std::deque<msg> messages;
    std::atomic<size_t> numBytes;
    std::atomic<unsigned> urgent;
    //! @todo FIXME track time since some epoch in an atomic.
    std::chrono::high_resolution_clock::time_point time;
    SimpleLock lock, timelock;

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
#ifndef NO_AGG
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
      auto n = std::chrono::high_resolution_clock::now();
      decltype(n) mytime;
      {
        std::lock_guard<SimpleLock> lg(timelock);
        mytime = time;
      }
      auto elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(n - mytime);
      if (elapsed.count() > COMM_DELAY) {
        ++statSendTimeout;
        return true;
      }
      return false;
#else
      return messages.size() > 0;
#endif
    }

    std::pair<uint32_t, std::vector<uint8_t>> assemble() {
      std::unique_lock<SimpleLock> lg(lock);
      if (messages.empty())
        return std::make_pair(~0, std::vector<uint8_t>());
#ifndef NO_AGG
      // compute message size
      uint32_t len = 0;
      int num      = 0;
      uint32_t tag = messages.front().tag;
      for (auto& m : messages) {
        if (m.tag != tag) {
          break;
        } else {
          // do not let it go over the integer limit because MPI_Isend cannot
          // deal with it
          if ((m.data.size() + sizeof(uint32_t) + len + num) >
              std::numeric_limits<int>::max()) {
            break;
          }
          len += m.data.size();
          num += sizeof(uint32_t);
        }
      }
      lg.unlock();
      // construct message
      std::vector<uint8_t> vec;
      vec.reserve(len + num);
      // go out of our way to avoid locking out senders when making messages
      lg.lock();
      do {
        auto& m = messages.front();
        lg.unlock();
        union {
          uint32_t a;
          uint8_t b[sizeof(uint32_t)];
        } foo;
        foo.a = m.data.size();
        vec.insert(vec.end(), &foo.b[0], &foo.b[sizeof(uint32_t)]);
        vec.insert(vec.end(), m.data.begin(), m.data.end());
        if (urgent)
          --urgent;
        lg.lock();
        messages.pop_front();
      } while (vec.size() < len + num);
      numBytes -= len;
#else
      uint32_t tag = messages.front().tag;
      std::vector<uint8_t> vec(std::move(messages.front().data));
      messages.pop_front();
#endif
      return std::make_pair(tag, std::move(vec));
    }

    void add(uint32_t tag, std::vector<uint8_t>& b) {
      std::lock_guard<SimpleLock> lg(lock);
      if (messages.empty()) {
        std::lock_guard<SimpleLock> lg(timelock);
        time = std::chrono::high_resolution_clock::now();
      }
      unsigned oldNumBytes = numBytes;
      numBytes += b.size();
      galois::runtime::trace("BufferedAdd", oldNumBytes, numBytes, tag,
                             galois::runtime::printVec(b));
      messages.emplace_back(tag, b);
    }
  }; // end send buffer class

  std::vector<sendBuffer> sendData;

  void workerThread() {
// Initialize LWCI or MPI depending on what was defined in CMake
#ifdef GALOIS_USE_LWCI
    // Initialize LWCI
    std::tie(netio, ID, Num) = makeNetworkIOLWCI(memUsageTracker);
    if (ID == 0)
      fprintf(stderr, "**Using LWCI Communication layer**\n");
#else
    initializeMPI();
    int rank;
    int hostSize;

    int rankSuccess = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rankSuccess != MPI_SUCCESS) {
      MPI_Abort(MPI_COMM_WORLD, rankSuccess);
    }

    int sizeSuccess = MPI_Comm_size(MPI_COMM_WORLD, &hostSize);
    if (sizeSuccess != MPI_SUCCESS) {
      MPI_Abort(MPI_COMM_WORLD, sizeSuccess);
    }

    galois::gDebug("[", NetworkInterface::ID, "] MPI initialized");
    std::tie(netio, ID, Num) = makeNetworkIOMPI(memUsageTracker);
#endif

    assert(ID == (unsigned)rank);
    assert(Num == (unsigned)hostSize);

    ready = 1;
    while (ready < 2) { /*fprintf(stderr, "[WaitOnReady-2]");*/
    };
    while (ready != 3) {
      for (unsigned i = 0; i < sendData.size(); ++i) {
        netio->progress();
        // handle send queue i
        auto& sd = sendData[i];
        if (sd.ready()) {
          NetworkIO::message msg;
          msg.host                    = i;
          std::tie(msg.tag, msg.data) = sd.assemble();
          galois::runtime::trace("BufferedSending", msg.host, msg.tag,
                                 galois::runtime::printVec(msg.data));
          ++statSendEnqueued;
          netio->enqueue(std::move(msg));
        }
        // handle receive
        NetworkIO::message rdata = netio->dequeue();
        if (rdata.data.size()) {
          ++statRecvDequeued;
          assert(rdata.data.size() !=
                 (unsigned int)std::count(rdata.data.begin(), rdata.data.end(),
                                          0));
          galois::runtime::trace("BufferedRecieving", rdata.host, rdata.tag,
                                 galois::runtime::printVec(rdata.data));
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
    ready  = 0;
    anyReceivedMessages = false;
    worker = std::thread(&NetworkInterfaceBuffered::workerThread, this);
    while (ready != 1) {
    };
    recvData = decltype(recvData)(Num);
    recvLock.resize(Num);
    sendData = decltype(sendData)(Num);
    ready    = 2;
  }

  virtual ~NetworkInterfaceBuffered() {
    ready = 3;
    worker.join();

// disable MPI if LWCI wasn't used
#ifndef GALOIS_USE_LWCI
    int finalizeSuccess = MPI_Finalize();

    if (finalizeSuccess != MPI_SUCCESS) {
      MPI_Abort(MPI_COMM_WORLD, finalizeSuccess);
    }

    galois::gDebug("[", NetworkInterface::ID, "] MPI finalized");
#endif
  }

  std::unique_ptr<galois::runtime::NetworkIO> netio;

  virtual void sendTagged(uint32_t dest, uint32_t tag, SendBuffer& buf) {
    statSendNum += 1;
    statSendBytes += buf.size();
    galois::runtime::trace("sendTagged", dest, tag,
                           galois::runtime::printVec(buf.getVec()));
    auto& sd = sendData[dest];
    sd.add(tag, buf.getVec());
  }

  virtual optional_t<std::pair<uint32_t, RecvBuffer>>
  recieveTagged(uint32_t tag,
                std::unique_lock<galois::substrate::SimpleLock>* rlg) {
    for (unsigned h = 0; h < recvData.size(); ++h) {
      auto& rq = recvData[h];
      if (rq.hasData(tag)) {
        if (recvLock[h].try_lock()) {
          std::unique_lock<galois::substrate::SimpleLock> lg(recvLock[h],
                                                             std::adopt_lock);
          auto buf = rq.popMsg(tag);
          if (buf) {
            ++statRecvNum;
            statRecvBytes += buf->size();
            memUsageTracker.decrementMemUsage(buf->size());
            if (rlg)
              *rlg = std::move(lg);
            galois::runtime::trace("recvTagged", h, tag,
                                   galois::runtime::printVec(buf->getVec()));
            anyReceivedMessages = true;
            return optional_t<std::pair<uint32_t, RecvBuffer>>(
                std::make_pair(h, std::move(*buf)));
          }
        }
      }
      galois::runtime::trace("recvTagged BLOCKED this by that", tag,
                             rq.getPresentTag());
#if 0
      else if (rq.getPresentTag() != ~0){
        galois::runtime::trace("recvTagged BLOCKED % by %", tag, rq.getPresentTag());
        if (recvLock[h].try_lock()) {
          std::unique_lock<galois::substrate::SimpleLock> lg(recvLock[h], std::adopt_lock);
          auto buf = rq.popMsg(rq.getPresentTag());
          if (buf) {
            if (rlg)
              *rlg = std::move(lg);
            uintptr_t fp = 0;
            gDeserializeRaw(buf->r_linearData() + buf->r_size() - sizeof(uintptr_t), fp);
            buf->pop_back(sizeof(uintptr_t));
            assert(fp);
            galois::runtime::trace("FP BLOCKED :", fp);
            return optional_t<std::pair<uint32_t, RecvBuffer>>();
          }
        }
      }
#endif
    }

    return optional_t<std::pair<uint32_t, RecvBuffer>>();
  }

  virtual void flush() {
    for (auto& sd : sendData)
      sd.markUrgent();
  }

  virtual bool anyPendingReceives() {
    if (anyReceivedMessages) { // might not be acted on by the computation yet
      anyReceivedMessages = false;
      return true;
    }
    for (unsigned h = 0; h < recvData.size(); ++h) {
      auto& rq = recvData[h];
      if (rq.size() > 0) return true;
    }
    return netio->anyPendingReceives();
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

  virtual std::vector<std::pair<std::string, unsigned long>>
  reportExtraNamed() const {
    std::vector<std::pair<std::string, unsigned long>> retval(5);
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

} // namespace

/**
 * Create a buffered network interface, or return one if already
 * created.
 */
NetworkInterface& galois::runtime::makeNetworkBuffered() {
  static std::atomic<NetworkInterfaceBuffered*> net;
  static substrate::SimpleLock m_mutex;

  // create the interface if it doesn't yet exist in the static variable
  auto* tmp = net.load();
  if (tmp == nullptr) {
    std::lock_guard<substrate::SimpleLock> lock(m_mutex);
    tmp = net.load();
    if (tmp == nullptr) {
      tmp = new NetworkInterfaceBuffered();
      net.store(tmp);
    }
  }

  return *tmp;
}
