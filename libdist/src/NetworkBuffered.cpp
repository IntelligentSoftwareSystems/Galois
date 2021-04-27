/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifdef GALOIS_USE_LCI
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

  // using vTy = std::vector<uint8_t>;
  using vTy = galois::PODResizeableArray<uint8_t>;

  static constexpr size_t kHeaderSize     = sizeof(BufferHeader);
  static constexpr uint8_t kMaxSegmentTag = std::numeric_limits<uint8_t>::max();
  static constexpr size_t kMaxBufferSize =
      static_cast<size_t>(std::numeric_limits<int>::max());
  static constexpr size_t kMaxDataSize = kMaxBufferSize - kHeaderSize;

  /**
   * Receive buffers for the buffered network interface
   */
  class recvBuffer {
    std::deque<NetworkIO::message> data;
    size_t frontOffset;
    SimpleLock qlock;
    // tag of head of queue
    std::atomic<uint32_t> dataPresent;

    struct PartialMessages {
      uint8_t num_segments{0};
      std::vector<vTy> segments;
    };
    std::unordered_map<uint8_t, PartialMessages> partial_messages_map_;

    std::optional<vTy> CombinePartialMessages(const BufferHeader& header,
                                              vTy&& vec) {
      auto& partial_messages = partial_messages_map_[header.segment_tag];
      if (partial_messages.num_segments == 0) {
        partial_messages.segments.resize(header.num_segments);
      }

      partial_messages.segments[header.segment_id] = std::move(vec);
      ++partial_messages.num_segments;

      if (partial_messages.num_segments != header.num_segments) {
        assert(partial_messages.num_segments < header.num_segments);
        assert(partial_messages.segments.size() == header.num_segments);
        return std::nullopt;
      }

      std::vector<vTy>& segments = partial_messages.segments;
      vTy message                = std::move(segments[0]);
      for (size_t i = 1, end = segments.size(); i < end; ++i) {
        message.insert(message.end(), segments[i].begin() + kHeaderSize,
                       segments[i].end());
      }
      partial_messages_map_.erase(header.segment_tag);
      return std::make_optional(std::move(message));
    }

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
    std::optional<vTy> popVec(uint32_t len,
                              std::atomic<size_t>& inflightRecvs) {
      if (data[0].data.size() == frontOffset + len) {
        vTy retval(std::move(data[0].data));
        data.pop_front();
        --inflightRecvs;
        frontOffset = 0;
        if (data.size()) {
          dataPresent = data.front().tag;
        } else {
          dataPresent = ~0;
        }
        return std::optional<vTy>(std::move(retval));
      } else {
        return std::optional<vTy>();
      }
    }

    void erase(size_t n, std::atomic<size_t>& inflightRecvs) {
      frontOffset += n;
      while (frontOffset && frontOffset >= data.front().data.size()) {
        frontOffset -= data.front().data.size();
        data.pop_front();
        --inflightRecvs;
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
    std::optional<RecvBuffer> popMsg(uint32_t tag,
                                     std::atomic<size_t>& inflightRecvs) {
      std::lock_guard<SimpleLock> lg(qlock);
      if (data.empty() || data.front().tag != tag)
        return std::optional<RecvBuffer>();

      vTy vec(std::move(data.front().data));

      data.pop_front();
      --inflightRecvs;
      if (!data.empty()) {
        dataPresent = data.front().tag;
      } else {
        dataPresent = ~0;
      }

      return std::optional<RecvBuffer>(RecvBuffer(std::move(vec), 0));
    }

    // Worker thread interface
    bool add(NetworkIO::message m) {
      BufferHeader* header = reinterpret_cast<BufferHeader*>(m.data.data());
      if (header->type == BufferHeader::BufferType::kPartialMessage) {
        std::optional<vTy> segment =
            CombinePartialMessages(*header, std::move(m.data));
        if (!segment) {
          return false;
        }

        m.data = std::move(*segment);
      }
      std::lock_guard<SimpleLock> lg(qlock);
      if (data.empty()) {
        galois::runtime::trace("ADD LATEST ", m.tag);
        dataPresent = m.tag;
      }

      data.push_back(std::move(m));
      return true;
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
      vTy data;
      msg(uint32_t t, vTy&& _data) : tag(t), data(std::move(_data)) {}
    };

    std::deque<msg> messages;
    std::atomic<size_t> numBytes;
    std::atomic<unsigned> urgent;
    //! @todo FIXME track time since some epoch in an atomic.
    std::chrono::high_resolution_clock::time_point time;
    SimpleLock lock, timelock;
    uint8_t segment_tag_{0};

    void IncrementSegmentTag() {
      if (segment_tag_ == kMaxSegmentTag) {
        segment_tag_ = 0;
      } else {
        ++segment_tag_;
      }
    }

    std::vector<NetworkIO::message> Split(uint32_t host, uint32_t tag,
                                          vTy&& vec) {
      std::vector<vTy> segments;
      segments.emplace_back(std::move(vec));
      auto begin = segments[0].begin();
      for (size_t i = kMaxBufferSize, end = segments[0].size(); i < end;
           i += kMaxDataSize) {
        vTy segment(kHeaderSize);
        size_t segment_end = std::min(end, i + kMaxDataSize);
        segment.insert(segment.end(), begin + i, begin + segment_end);
        segments.emplace_back(std::move(segment));
      }
      segments[0].resize(kMaxBufferSize);

      std::vector<NetworkIO::message> msg;
      for (size_t i = 0; i < segments.size(); ++i) {
        auto& segment        = segments[i];
        BufferHeader* header = reinterpret_cast<BufferHeader*>(segment.data());
        header->type         = BufferHeader::BufferType::kPartialMessage;
        header->num_segments = segments.size();
        header->segment_id   = i;
        header->segment_tag  = segment_tag_;
        msg.emplace_back(host, tag, std::move(segment));
      }
      IncrementSegmentTag();
      return msg;
    }

  public:
    unsigned long statSendTimeout;
    unsigned long statSendOverflow;
    unsigned long statSendUrgent;

    size_t size() { return messages.size(); }

    void markUrgent() {
      if (numBytes) {
        std::lock_guard<SimpleLock> lg(lock);
        urgent = messages.size();
      }
    }

    bool ready() { return messages.size() > 0; }

    std::vector<NetworkIO::message> assemble(uint32_t host) {
      std::unique_lock<SimpleLock> lg(lock);
      assert(!messages.empty());
      uint32_t tag = messages.front().tag;
      vTy vec(std::move(messages.front().data));
      messages.pop_front();

      if (vec.size() > kMaxBufferSize) {
        return Split(host, tag, std::move(vec));
      }

      BufferHeader* header = reinterpret_cast<BufferHeader*>(vec.data());
      header->type         = BufferHeader::BufferType::kSingleMessage;
      std::vector<NetworkIO::message> msgs;
      msgs.emplace_back(host, tag, std::move(vec));
      return msgs;
    }

    void add(uint32_t tag, vTy&& b) {
      std::lock_guard<SimpleLock> lg(lock);
      if (messages.empty()) {
        std::lock_guard<SimpleLock> lg(timelock);
        time = std::chrono::high_resolution_clock::now();
      }
      assert(b.size() >= kHeaderSize);
      numBytes += b.size();
      messages.emplace_back(tag, std::move(b));
    }
  }; // end send buffer class

  std::vector<sendBuffer> sendData;

  void workerThread() {
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
    std::tie(netio, ID, Num) =
        makeNetworkIOMPI(memUsageTracker, inflightSends, inflightRecvs);

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
          std::vector<NetworkIO::message> msgs = sd.assemble(i);
          if (msgs.size() > 1) {
            inflightSends += msgs.size() - 1;
          }

          for (auto& msg : msgs) {
            ++statSendEnqueued;
            netio->enqueue(std::move(msg));
          }
        }

        // handle receive
        NetworkIO::message rdata = netio->dequeue();
        if (rdata.data.size()) {
          ++statRecvDequeued;
          uint32_t h               = rdata.host;
          bool not_partial_segment = recvData[h].add(std::move(rdata));
          if (!not_partial_segment) {
            --inflightRecvs;
          }
        }
      }
    }
    finalizeMPI();
  }

  std::thread worker;
  std::atomic<int> ready;

public:
  using NetworkInterface::ID;
  using NetworkInterface::Num;

  NetworkInterfaceBuffered() {
    inflightSends       = 0;
    inflightRecvs       = 0;
    ready               = 0;
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
  }

  std::unique_ptr<galois::runtime::NetworkIO> netio;

  virtual void sendTagged(uint32_t dest, uint32_t tag, SendBuffer&& buf,
                          int phase) {
    tag += phase;
    statSendNum += 1;
    statSendBytes += buf.size() + kHeaderSize;
    memUsageTracker.incrementMemUsage(buf.size() + kHeaderSize);
    ++inflightSends;
    auto& sd = sendData[dest];
    sd.add(tag, std::move(buf.get()));
  }

  virtual std::optional<std::pair<uint32_t, RecvBuffer>>
  recieveTagged(uint32_t tag, int phase) {
    tag += phase;
    for (unsigned h = 0; h < recvData.size(); ++h) {
      auto& rq = recvData[h];
      if (rq.hasData(tag)) {
        if (recvLock[h].try_lock()) {
          std::unique_lock<galois::substrate::SimpleLock> lg(recvLock[h],
                                                             std::adopt_lock);
          auto buf = rq.popMsg(tag, inflightRecvs);
          if (buf) {
            ++statRecvNum;
            statRecvBytes += buf->size() + kHeaderSize;
            memUsageTracker.decrementMemUsage(buf->size() + kHeaderSize);
            anyReceivedMessages = true;
            return std::optional<std::pair<uint32_t, RecvBuffer>>(
                std::make_pair(h, std::move(*buf)));
          }
        }
      }
      galois::runtime::trace("recvTagged BLOCKED this by that", tag,
                             rq.getPresentTag());
    }

    return std::optional<std::pair<uint32_t, RecvBuffer>>();
  }

  virtual void flush() {
    for (auto& sd : sendData)
      sd.markUrgent();
  }

  virtual bool anyPendingSends() { return (inflightSends > 0); }

  virtual bool anyPendingReceives() {
    if (anyReceivedMessages) { // might not be acted on by the computation yet
      anyReceivedMessages = false;
      // galois::gDebug("[", ID, "] receive out of buffer \n");
      return true;
    }
    // if (inflightRecvs > 0) {
    // galois::gDebug("[", ID, "] inflight receive: ", inflightRecvs, " \n");
    // }
    return (inflightRecvs > 0);
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
