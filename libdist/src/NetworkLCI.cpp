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
 * Contains NetworkInterfaceLCI, an implementation of a network interface
 * that buffers messages before sending them out.
 *
 * @todo document this file more
 */

#ifdef GALOIS_USE_LWCI
#include "galois/runtime/Network.h"
#include "galois/runtime/NetworkIO.h"
#include "galois/runtime/Tracer.h"
#include "galois/runtime/LWCI.h"

using vTy = galois::PODResizeableArray<uint8_t>;

#include <thread>
#include <mutex>
#include <iostream>
#include <limits>
#include <queue>

#include <boost/lockfree/queue.hpp>

using namespace galois::runtime;
using namespace galois::substrate;

/* CRC-32C (iSCSI) polynomial in reversed bit order. */
#define POLY 0x82f63b78
inline uint32_t crc32c(char* buf, size_t len) {
  uint32_t crc = 0;
  int k;

  crc = ~crc;
  while (len--) {
    crc ^= *buf++;
    for (k = 0; k < 8; k++)
      crc = crc & 1 ? (crc >> 1) ^ POLY : crc >> 1;
  }
  return ~crc;
}

lc_ep lc_p2p_ep[3];
lc_ep lc_col_ep;

struct pendingReq {
  uint32_t dest;
  uint32_t tag;
  int phase;
  vTy buf;
  lc_req req;
  std::atomic<size_t>& inflight;
  pendingReq(uint32_t _d, uint32_t _t, int _p, vTy& _buf,
             std::atomic<size_t>& s)
      : dest(_d), tag(_t), phase(_p), buf(std::move(_buf)), inflight(s) {
    s++;
  }
  ~pendingReq() { inflight--; }
};

static void* alloc_req(size_t size, void** ctx) {
  vTy** vector = (vTy**)ctx;
  *vector      = new vTy(size);
  return (*vector)->data();
}

static void free_req(void* ctx) {
  pendingReq* req = (pendingReq*)ctx;
  delete req;
}

namespace {

/**
 * @class NetworkInterfaceLCI
 *
 * Buffered network interface: messages are buffered before they are sent out.
 * A single worker thread is initialized to send/receive messages from/to
 * buffers.
 */
class NetworkInterfaceLCI : public NetworkInterface {
  unsigned long statSendNum;
  unsigned long statSendBytes;
  unsigned long statSendEnqueued;
  unsigned long statRecvNum;
  unsigned long statRecvBytes;
  unsigned long statRecvDequeued;
  bool anyReceivedMessages;

  // using vTy = std::vector<uint8_t>;
  using vTy = galois::PODResizeableArray<uint8_t>;

public:
  void workerThread() {
    // Initialize LWCI
    // makeNetworkIOLWCI(memUsageTracker, inflightSends, inflightRecvs);
    if (ID == 0)
      fprintf(stderr, "**Using LWCI Communication layer**\n");

    ready = 1;
    while (ready < 2) { /*fprintf(stderr, "[WaitOnReady-2]");*/
    };
    while (ready != 3) {
      lc_progress(0);

      lc_req* req_ptr;
      for (int phase = 0; phase < 3; phase++) {
        if (lc_cq_pop(lc_p2p_ep[phase], &req_ptr) == LC_OK) {
          int bin = ((req_ptr->meta % 3) * 3) + phase;
          bufferedRecv[bin].push(convertReq(req_ptr, phase));
        }
      }

      sched_yield();
    }
  }

  std::thread worker;
  std::atomic<int> ready;

public:
  using NetworkInterface::ID;
  using NetworkInterface::Num;

  NetworkInterfaceLCI() {
    lc_init(1, &lc_col_ep);
    lc_opt opt;
    opt.dev   = 0;
    opt.desc  = LC_DYN_CQ;
    opt.alloc = alloc_req;
    lc_ep_dup(&opt, lc_col_ep, &lc_p2p_ep[0]);
    lc_ep_dup(&opt, lc_col_ep, &lc_p2p_ep[1]);
    lc_ep_dup(&opt, lc_col_ep, &lc_p2p_ep[2]);

    lc_get_proc_num((int*)&ID);
    lc_get_num_proc((int*)&Num);

    inflightSends       = 0;
    inflightRecvs       = 0;
    ready               = 0;
    anyReceivedMessages = false;
    worker              = std::thread(&NetworkInterfaceLCI::workerThread, this);
    while (ready != 1) {
    };
    ready = 2;
  }

  virtual ~NetworkInterfaceLCI() {
    ready = 3;
    worker.join();
  }

  boost::lockfree::queue<pendingReq*>
      bufferedRecv[9]; // [0, 1, 2] [0, 1, 2] 0: normal, 1: reduce, 2: AM

  virtual void sendTagged(uint32_t dest, uint32_t tag, SendBuffer& buf,
                          int phase) {
    if (tag == 0)
      phase = 2;

    statSendNum += 1;
    statSendBytes += buf.size();
    // int count = 0;
#ifndef __GALOIS_HET_ASYNC__
    if (buf.getVec().size() < 8192) {
      while (lc_sendm(buf.getVec().data(), buf.getVec().size(), dest, tag,
                      lc_p2p_ep[phase]) != LC_OK) {
        sched_yield();
      }
    } else
#endif
    {
      pendingReq* msg =
          new pendingReq(dest, tag, phase, buf.getVec(), inflightSends);
      while (lc_sendl(msg->buf.data(), msg->buf.size(), dest, tag,
                      lc_p2p_ep[phase], free_req, msg) != LC_OK) {
        sched_yield();
      }
    }
  }

  inline pendingReq* convertReq(lc_req* req_ptr, int phase) {
    // Need to drain LCI queue to allow more injection.
    // Convert internal LCI request to a Galois pending request.
    vTy buf  = std::move(*((vTy*)(req_ptr->ctx)));
    int rank = req_ptr->rank;
    int meta = req_ptr->meta;
    delete (vTy*)req_ptr->ctx;
    lc_cq_reqfree(lc_p2p_ep[phase], req_ptr);
    return new pendingReq(rank, meta, phase, buf, inflightRecvs);
  }

  virtual optional_t<std::pair<uint32_t, RecvBuffer>>
  recieveTagged(uint32_t tag,
                std::unique_lock<galois::substrate::SimpleLock>* rlg,
                int phase) {
    if (tag == 0)
      phase = 2;
    // static int count = 0;

    pendingReq* req;
    int bin = ((tag % 3) * 3) + phase;
    if (!bufferedRecv[bin].pop(req)) {
      // if (count ++ == 10000) {
      //  printf("[%d] WARNING possible lock out on RECV %d\n", ID, tag);
      // }
      return optional_t<std::pair<uint32_t, RecvBuffer>>();
    }

    if (req->tag == tag) {
      vTy buf  = std::move(req->buf);
      int dest = req->dest;
      delete req;
      return optional_t<std::pair<uint32_t, RecvBuffer>>(
          std::make_pair(dest, std::move(buf)));
    } else {
      printf("[%d] WARNING possible lock out, wrong tag %d/%d.\n", ID, req->tag,
             tag);
      return optional_t<std::pair<uint32_t, RecvBuffer>>();
    }
  }

  virtual void flush() {}

  virtual bool anyPendingSends() {
    // static int count = 0;
    // if (count++ == 10000)
    // printf("[%d] WARNING possible lock out terminate %d %d\n", ID,
    // inflightSends.load(), inflightRecvs.load());
    return (inflightSends > 0);
  }

  virtual bool anyPendingReceives() {
    if (anyReceivedMessages) { // might not be acted on by the computation yet
      anyReceivedMessages = false;
      // galois::gDebug("[", ID, "] receive out of buffer \n");
      return true;
    }
    galois::gDebug("[", ID, "] inflight receive: ", inflightRecvs, " \n");
    return (inflightRecvs > 0);
  }

  virtual unsigned long reportSendBytes() const { return statSendBytes; }
  virtual unsigned long reportSendMsgs() const { return statSendNum; }
  virtual unsigned long reportRecvBytes() const { return statRecvBytes; }
  virtual unsigned long reportRecvMsgs() const { return statRecvNum; }

  virtual std::vector<unsigned long> reportExtra() const {
    std::vector<unsigned long> retval(5);
    return retval;
  }

  virtual std::vector<std::pair<std::string, unsigned long>>
  reportExtraNamed() const {
    std::vector<std::pair<std::string, unsigned long>> retval(5);
    retval[0].first  = "SendTimeout";
    retval[1].first  = "SendOverflow";
    retval[2].first  = "SendUrgent";
    retval[3].first  = "SendEnqueued";
    retval[4].first  = "RecvDequeued";
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
NetworkInterface& galois::runtime::makeNetworkLCI() {
  static std::atomic<NetworkInterfaceLCI*> net;
  static substrate::SimpleLock m_mutex;

  // create the interface if it doesn't yet exist in the static variable
  auto* tmp = net.load();
  if (tmp == nullptr) {
    std::lock_guard<substrate::SimpleLock> lock(m_mutex);
    tmp = net.load();
    if (tmp == nullptr) {
      tmp = new NetworkInterfaceLCI();
      net.store(tmp);
    }
  }

  return *tmp;
}
#endif
