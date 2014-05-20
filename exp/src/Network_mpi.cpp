/** Galois Network Layer for MPI -*- C++ -*-
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
 * @author Manoj Dhanapal <madhanap@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */


#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/NetworkBackend.h"
#include "Galois/Runtime/Tracer.h"
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/TID.h"

#include <mpi.h>

#include <deque>
#include <utility>

using namespace Galois::Runtime;

#define INFLIGHT_LIMIT 100000

namespace {

static void handleError(int rc) {
  if (rc != MPI_SUCCESS) {
    GALOIS_ERROR(false, "MPI ERROR"); 
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
}

static const int FuncTag = 1;

class NetworkInterfaceAsyncMPI : public NetworkInterface {

  Galois::Runtime::LL::SimpleLock lock;
  std::deque<std::pair<MPI_Request, SendBuffer>> pending_sends;

  Galois::optional<RecvBuffer> doOneRecv() {
    std::lock_guard<decltype(lock)> lg(lock);
    update_pending_sends();
    Galois::optional<RecvBuffer> retval;
    MPI_Status status;
    //async probe
    int flag, rv;
    rv = MPI_Iprobe(MPI_ANY_SOURCE, FuncTag, MPI_COMM_WORLD, &flag, &status);
    handleError(rv);
    if (flag) {
      int count;
      MPI_Get_count(&status, MPI_BYTE, &count);
      retval = RecvBuffer(count);
      MPI_Recv(retval->linearData(), count, MPI_BYTE, MPI_ANY_SOURCE, FuncTag, MPI_COMM_WORLD, &status);
    }
    return retval;
  }

  void update_pending_sends() {
    int flag = true;
    while (flag && !pending_sends.empty()) {
      MPI_Status s;
      int rv = MPI_Test(&pending_sends.front().first, &flag, &s);
      handleError(rv);
      if (flag)
        pending_sends.pop_front();
    }
  }

public:

  using NetworkInterface::ID;
  using NetworkInterface::Num;


  NetworkInterfaceAsyncMPI() {
    int provided;
    int rc = MPI_Init_thread (NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    handleError(rc);

    int numTasks, taskRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskRank);
    
    ID = taskRank;
    Num = numTasks;
    
    if (taskRank == 0) {
      //master
      //printing on lead host doesn't require this object to be fully initialized
      switch (provided) {
      case MPI_THREAD_SINGLE: 
      case MPI_THREAD_FUNNELED: 
        assert(0 && "Insufficient mpi support");
        abort();
        break;
      case MPI_THREAD_SERIALIZED: Galois::Runtime::LL::gInfo("MPI Supports: Serialized"); break;
      case MPI_THREAD_MULTIPLE: Galois::Runtime::LL::gInfo("MPI Supports: Multiple"); break;
      default: break;
      }
    } else {
      //slave
    }
  }

  virtual ~NetworkInterfaceAsyncMPI() {
    MPI_Finalize();
  }

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    //trace("NetworkInterfaceAsyncMPI::send % to %\n", (void*)recv, dest);
    lock.lock();
    //wait for a send slot
    // while (pending_sends.size() >= 128)
    //   update_pending_sends();
    assert(recv);
    buf.serialize_header((void*)recv);
    //trace("NetworkInterfaceAsyncMPI::send buf %\n", buf);
    assert(dest < Num);
    pending_sends.emplace_back(MPI_REQUEST_NULL, std::move(buf));
    auto & com = pending_sends.back();
    int rv = MPI_Issend(com.second.linearData(), com.second.size(), MPI_BYTE, dest, FuncTag, MPI_COMM_WORLD, &com.first);
    handleError(rv);
    update_pending_sends();
    lock.unlock();
  }

  virtual bool handleReceives() {
    Galois::optional<RecvBuffer> data;
    bool retval = false;
    while ((data = doOneRecv())) {
      retval = true;
      //Unlocked call so reciever can call handleRecv()
      recvFuncTy f;
      uintptr_t fp;
      gDeserialize(*data,fp);
      assert(fp);
      f = (recvFuncTy)fp;
      //trace("NetworkInterfaceAsyncMPI::handleRecieves %\n", (void*)f);
      //Call deserialized function
      f(*data);
    }
    return retval;
  }

  virtual bool needsDedicatedThread() {
    return true;
  }

  virtual void systemBarrier() {
    int rv;
    rv = MPI_Barrier(MPI_COMM_WORLD);
    handleError(rv);
    return;
  }

};

static const int numslots = 16;
class NetworkBackendMPI : public NetworkBackend {

  LL::SimpleLock lock;
  std::pair<MPI_Request, SendBlock*> pending_sends[numslots];
  int pending_start;
  int pending_end;
  BlockList waiting;

  //! requires lock be held
  //! clear completed requests from the queue
  void clear_sends() {
    int flag = true;
    while (pending_start != pending_end && flag) {
      MPI_Status s;
      int rv = MPI_Test(&pending_sends[pending_start].first, &flag, &s);
      handleError(rv);
      if (flag) {
        freeSendBlock(pending_sends[pending_start].second);
        pending_sends[pending_start].second = nullptr;
        pending_start = (pending_start + 1) % numslots;
      }
    }
  }
 
  //! requires lock be held
  //! if slots allow, initialize more sends
  void do_more_sends() {
    while (!waiting.empty() && ((pending_end + 1) % numslots) != pending_start) {
      auto& com = pending_sends[pending_end];
      com.first = MPI_REQUEST_NULL;
      com.second = &waiting.front();
      waiting.pop_front();
      pending_end = (pending_end + 1) % numslots;
      int rv = MPI_Isend(com.second->data, com.second->size, MPI_BYTE, com.second->dest, FuncTag, MPI_COMM_WORLD, &com.first);
      handleError(rv);
    }
  }

    void update_pending_sends() {
      clear_sends();
      do_more_sends();
      clear_sends();
    }

  static const unsigned msgSize = 1024*4;

public:

  NetworkBackendMPI() : NetworkBackend(msgSize) {
    int provided;
    int rc = MPI_Init_thread (NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    handleError(rc);

    int numTasks, taskRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskRank);
    
    NetworkBackend::_ID  = taskRank;
    NetworkBackend::_Num = numTasks;
    
    if (taskRank == 0) {
      //master
      //printing on lead host doesn't require this object to be fully initialized
      switch (provided) {
      case MPI_THREAD_SINGLE: 
      case MPI_THREAD_FUNNELED:
      default:
        assert(0 && "Insufficient mpi support");
        abort();
        break;
      case MPI_THREAD_SERIALIZED: Galois::Runtime::LL::gInfo("MPI Supports: Serialized"); break;
      case MPI_THREAD_MULTIPLE: Galois::Runtime::LL::gInfo("MPI Supports: Multiple"); break;
      }
    }
  }

  virtual ~NetworkBackendMPI() {
    MPI_Finalize();
  }

  virtual void send(SendBlock* data) {
    std::lock_guard<decltype(lock)> lg(lock);
    waiting.push_back(*data);
    update_pending_sends();
  }

  virtual SendBlock* recv() {
    std::lock_guard<decltype(lock)> lg(lock);
    update_pending_sends();
    SendBlock* retval = nullptr;
    MPI_Status status;
    //async probe
    int flag, rv;
    rv = MPI_Iprobe(MPI_ANY_SOURCE, FuncTag, MPI_COMM_WORLD, &flag, &status);
    handleError(rv);
    if (flag) {
      int count = 0;
      MPI_Get_count(&status, MPI_BYTE, &count);
      retval = allocSendBlock();
      rv = MPI_Recv(retval->data, count, MPI_BYTE, MPI_ANY_SOURCE, FuncTag, MPI_COMM_WORLD, &status);
      retval->size = count;
      retval->dest = status.MPI_SOURCE;
      handleError(rv);
    }
    return retval;
  }

  virtual void flush(bool block) {
    bool cont = true;
    do {
      std::lock_guard<decltype(lock)> lg(lock);
      update_pending_sends();
      cont = !waiting.empty();
      //This order let's us release the lock for a moment
    } while (block && cont);
  }

};

}

#ifdef USE_MPI
NetworkInterface& Galois::Runtime::getSystemNetworkInterface() {
  static NetworkInterfaceAsyncMPI net;
  return net;
}
#endif

NetworkBackend& Galois::Runtime::getSystemNetworkBackend() {
  static NetworkBackendMPI net;
  return net;
}
