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
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/ll/TID.h"

#include <mpi.h>

#include <deque>
#include <utility>

using namespace Galois::Runtime;

#define INFLIGHT_LIMIT 100000

bool Galois::Runtime::inDoAllDistributed = false;

namespace {

static void handleError(int rc) {
  if (rc != MPI_SUCCESS) {
    GALOIS_ERROR(false, "MPI ERROR"); 
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
}

class MPIBase {

  int numTasks, taskRank;

  Galois::Runtime::LL::SimpleLock<true> lock;

  // const int FuncTag = 1;
  #define FuncTag 1

protected:
  std::deque<std::pair<MPI_Request, SendBuffer>> pending_sends;

public:
  MPIBase() {
    int provided;
    int rc = MPI_Init_thread (NULL, NULL, MPI_THREAD_SERIALIZED /*MPI_THREAD_FUNNELED*/, &provided);
    handleError(rc);
    
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskRank);

    // std::cout << "I am " << taskRank << " of " << numTasks << "\n";

    networkHostID = taskRank;
    networkHostNum = numTasks;

    if (taskRank == 0) {
      //master
      //printing on lead host doesn't require this object to be fully initialized
      switch (provided) {
      case MPI_THREAD_SINGLE: Galois::Runtime::LL::gInfo("MPI Supports: Single"); break;
      case MPI_THREAD_FUNNELED: Galois::Runtime::LL::gInfo("MPI Supports: Funneled"); break;
      case MPI_THREAD_SERIALIZED: Galois::Runtime::LL::gInfo("MPI Supports: Serialized"); break;
      case MPI_THREAD_MULTIPLE: Galois::Runtime::LL::gInfo("MPI Supports: Multiple"); break;
      default: break;
      }
    } else {
      //slave
    }
  }

  ~MPIBase() {
    MPI_Finalize();
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

  //! sends a message.  assumes it is being called from a thread for which
  //! this is valid
  void sendInternal(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    assert(Galois::Runtime::LL::getTID() == 0);
    assert(recv);
    buf.serialize_header((void*)recv);
    if (true) {
      pending_sends.emplace_back(MPI_REQUEST_NULL, std::move(buf));
      std::pair<MPI_Request, SendBuffer>& com = pending_sends.back();
      assert(com.second.size());
      int rv = MPI_Issend(com.second.linearData(), com.second.size(), MPI_BYTE, dest, FuncTag, MPI_COMM_WORLD, &com.first);
      handleError(rv);
    } else {
      assert(dest < networkHostNum);
      int rv = MPI_Send(buf.linearData(), buf.size(), MPI_BYTE, dest, FuncTag, MPI_COMM_WORLD);
      handleError(rv);
    }
  }

  bool recvInternal() {
    assert(Galois::Runtime::LL::getTID() == 0);
    int flag, rv;
    bool retval = false;
    MPI_Status status;
    do {
      //async probe
      rv = MPI_Iprobe(MPI_ANY_SOURCE, FuncTag, MPI_COMM_WORLD, &flag, &status);
      handleError(rv);
      if (!flag)
        break;
      //use the lock
      lock.lock();
      rv = MPI_Iprobe(MPI_ANY_SOURCE, FuncTag, MPI_COMM_WORLD, &flag, &status);
      handleError(rv);
      if (flag) {
        retval = true;
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        RecvBuffer buf(count);
        MPI_Recv(buf.linearData(), count, MPI_BYTE, MPI_ANY_SOURCE, FuncTag, MPI_COMM_WORLD, &status);
        lock.unlock();
        //Unlocked call so reciever can call handleRecv()
        recvFuncTy f;
        uintptr_t fp;
        gDeserialize(buf,fp);
        assert(fp);
        f = (recvFuncTy)fp;
        //Call deserialized function
        f(buf);
      } else {
        lock.unlock();
      }
    } while (true);
    return retval;
  }
};

//! handle mpi implementations which require a single thread to communicate
class NetworkInterfaceSyncMPI : public NetworkInterface, public MPIBase {

  struct outgoingMessage {
    uint32_t dest;
    recvFuncTy recv;
    SendBuffer buf;
    outgoingMessage(uint32_t _dest, recvFuncTy _recv, SendBuffer& _buf)
      :dest(_dest), recv(_recv), buf(std::move(_buf))
    {}

  };
  std::deque<outgoingMessage> asyncOutQueue;
  Galois::Runtime::LL::SimpleLock<true> asyncOutLock;

public:
  virtual ~NetworkInterfaceSyncMPI() {
  }

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    asyncOutLock.lock();
    if (Galois::Runtime::LL::getTID() == 0) {
      update_pending_sends();
      while (!asyncOutQueue.empty() && (pending_sends.size() < INFLIGHT_LIMIT)) {
        sendInternal(asyncOutQueue[0].dest, asyncOutQueue[0].recv, asyncOutQueue[0].buf);
        asyncOutQueue.pop_front();
        update_pending_sends();
      }
      if (!asyncOutQueue.empty() || (pending_sends.size() >= INFLIGHT_LIMIT))
       asyncOutQueue.emplace_back(dest, recv, buf);
      else
       sendInternal(dest, recv, buf);
    } else {
      asyncOutQueue.emplace_back(dest, recv, buf);
    }
    asyncOutLock.unlock();
  }

  virtual void systemBarrier() {
    int rv;
    rv = MPI_Barrier(MPI_COMM_WORLD);
    handleError(rv);
    return;
  }

  virtual bool handleReceives() {
    assert(Galois::Runtime::LL::getTID() == 0);
    bool retval = recvInternal();

    asyncOutLock.lock();
    update_pending_sends();
    while (!asyncOutQueue.empty() && (pending_sends.size() < INFLIGHT_LIMIT)) {
      sendInternal(asyncOutQueue[0].dest, asyncOutQueue[0].recv, asyncOutQueue[0].buf);
      asyncOutQueue.pop_front();
      update_pending_sends();
    }
    asyncOutLock.unlock();

    return retval;
  }

  virtual bool needsDedicatedThread() {
    return true;
  }

};

//! handle mpi implementatinos which are multithreaded
class NetworkInterfaceAsyncMPI : public NetworkInterface, public MPIBase {

public:
  virtual ~NetworkInterfaceAsyncMPI() {
  }

  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    sendInternal(dest, recv, buf);
  }

  virtual bool handleReceives() {
    return recvInternal();
  }

  virtual bool needsDedicatedThread() {
    return false;
  }

  virtual void systemBarrier() {
    return;
  }

};

}


NetworkInterface& Galois::Runtime::getSystemNetworkInterface() {
  static NetworkInterfaceSyncMPI net;
  return net;
}
