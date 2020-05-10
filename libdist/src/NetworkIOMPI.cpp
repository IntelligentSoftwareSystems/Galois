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
 * @file NetworkIOMPI.cpp
 *
 * Contains an implementation of network IO that uses MPI.
 */

#include "galois/runtime/NetworkIO.h"
#include "galois/runtime/Tracer.h"
#include "galois/substrate/SimpleLock.h"

/**
 * MPI implementation of network IO. ASSUMES THAT MPI IS INITIALIZED
 * UPON CREATION OF THIS OBJECT.
 */
class NetworkIOMPI : public galois::runtime::NetworkIO {
private:
  /**
   * Get the host id of the caller.
   *
   * @returns host id of the caller with regard to the MPI setup
   */
  static int getID() {
    int taskRank;
    handleError(MPI_Comm_rank(MPI_COMM_WORLD, &taskRank));
    return taskRank;
  }

  /**
   * Get the total number of hosts in the system.
   *
   * @returns number of hosts with regard to the MPI setup
   */
  static int getNum() {
    int numTasks;
    handleError(MPI_Comm_size(MPI_COMM_WORLD, &numTasks));
    return numTasks;
  }

  /**
   * Get both the ID of the caller + number of hosts.
   */
  std::pair<int, int> getIDAndHostNum() {
    return std::make_pair(getID(), getNum());
  }

  /**
   * Message type to send/recv in this network IO layer.
   */
  struct mpiMessage {
    uint32_t host;
    uint32_t tag;
    vTy data;
    MPI_Request req;
    // mpiMessage(message&& _m, MPI_Request _req) : m(std::move(_m)), req(_req)
    // {}
    mpiMessage(uint32_t host, uint32_t tag, vTy&& data)
        : host(host), tag(tag), data(std::move(data)) {}
    mpiMessage(uint32_t host, uint32_t tag, size_t len)
        : host(host), tag(tag), data(len) {}
  };

  /**
   * Send queue structure.
   */
  struct sendQueueTy {
    std::deque<mpiMessage> inflight;

    galois::runtime::MemUsageTracker& memUsageTracker;

    std::atomic<size_t>& inflightSends;

    sendQueueTy(galois::runtime::MemUsageTracker& tracker,
                std::atomic<size_t>& sends)
        : memUsageTracker(tracker), inflightSends(sends) {}

    void complete() {
      while (!inflight.empty()) {
        int flag = 0;
        MPI_Status status;
        auto& f = inflight.front();
        int rv  = MPI_Test(&f.req, &flag, &status);
        handleError(rv);
        if (flag) {
          memUsageTracker.decrementMemUsage(f.data.size());
          inflight.pop_front();
          --inflightSends;
        } else
          break;
      }
    }

    void send(message m) {
      inflight.emplace_back(m.host, m.tag, std::move(m.data));
      auto& f = inflight.back();
      galois::runtime::trace("MPI SEND", f.host, f.tag, f.data.size(),
                             galois::runtime::printVec(f.data));
#ifdef GALOIS_SUPPORT_ASYNC
      int rv = MPI_Issend(f.data.data(), f.data.size(), MPI_BYTE, f.host, f.tag,
                          MPI_COMM_WORLD, &f.req);
#else
      int rv = MPI_Isend(f.data.data(), f.data.size(), MPI_BYTE, f.host, f.tag,
                         MPI_COMM_WORLD, &f.req);
#endif
      handleError(rv);
    }
  };

  /**
   * Receive queue structure
   */
  struct recvQueueTy {
    std::deque<message> done;
    std::deque<mpiMessage> inflight;

    galois::runtime::MemUsageTracker& memUsageTracker;

    std::atomic<size_t>& inflightRecvs;

    recvQueueTy(galois::runtime::MemUsageTracker& tracker,
                std::atomic<size_t>& recvs)
        : memUsageTracker(tracker), inflightRecvs(recvs) {}

    // FIXME: Does synchronous recieves overly halt forward progress?
    void probe() {
      int flag = 0;
      MPI_Status status;
      // check for new messages
      int rv = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag,
                          &status);
      handleError(rv);
      if (flag) {
        ++inflightRecvs;
        int nbytes;
        rv = MPI_Get_count(&status, MPI_BYTE, &nbytes);
        handleError(rv);
#ifdef GALOIS_USE_BARE_MPI
        assert(status.MPI_TAG <= 32767);
        if (status.MPI_TAG != 32767) {
#endif
          inflight.emplace_back(status.MPI_SOURCE, status.MPI_TAG, nbytes);
          auto& m = inflight.back();
          memUsageTracker.incrementMemUsage(m.data.size());
          rv = MPI_Irecv(m.data.data(), nbytes, MPI_BYTE, status.MPI_SOURCE,
                         status.MPI_TAG, MPI_COMM_WORLD, &m.req);
          handleError(rv);
          galois::runtime::trace("MPI IRECV", status.MPI_SOURCE, status.MPI_TAG,
                                 m.data.size());
#ifdef GALOIS_USE_BARE_MPI
        }
#endif
      }

      // complete messages
      if (!inflight.empty()) {
        auto& m  = inflight.front();
        int flag = 0;
        rv       = MPI_Test(&m.req, &flag, MPI_STATUS_IGNORE);
        handleError(rv);
        if (flag) {
          done.emplace_back(m.host, m.tag, std::move(m.data));
          inflight.pop_front();
        }
      }
    }
  };

  sendQueueTy sendQueue;
  recvQueueTy recvQueue;

public:
  /**
   * Constructor.
   *
   * @param tracker memory usage tracker
   * @param [out] ID this machine's host id
   * @param [out] NUM total number of hosts in the system
   */
  NetworkIOMPI(galois::runtime::MemUsageTracker& tracker,
               std::atomic<size_t>& sends, std::atomic<size_t>& recvs,
               uint32_t& ID, uint32_t& NUM)
      : NetworkIO(tracker, sends, recvs), sendQueue(tracker, inflightSends),
        recvQueue(tracker, inflightRecvs) {
    auto p = getIDAndHostNum();
    ID     = p.first;
    NUM    = p.second;
  }

  /**
   * Adds a message to the send queue
   */
  virtual void enqueue(message m) {
    memUsageTracker.incrementMemUsage(m.data.size());
    sendQueue.send(std::move(m));
  }

  /**
   * Attempts to get a message from the recv queue.
   */
  virtual message dequeue() {
    if (!recvQueue.done.empty()) {
      auto msg = std::move(recvQueue.done.front());
      recvQueue.done.pop_front();
      return msg;
    }
    return message{~0U, 0, vTy()};
  }

  /**
   * Push progress forward in the system.
   */
  virtual void progress() {
    sendQueue.complete();
    recvQueue.probe();
  }
}; // end NetworkIOMPI class

std::tuple<std::unique_ptr<galois::runtime::NetworkIO>, uint32_t, uint32_t>
galois::runtime::makeNetworkIOMPI(galois::runtime::MemUsageTracker& tracker,
                                  std::atomic<size_t>& sends,
                                  std::atomic<size_t>& recvs) {
  uint32_t ID, NUM;
  std::unique_ptr<galois::runtime::NetworkIO> n{
      new NetworkIOMPI(tracker, sends, recvs, ID, NUM)};
  return std::make_tuple(std::move(n), ID, NUM);
}
