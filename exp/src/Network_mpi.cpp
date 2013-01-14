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

#include <mpi.h>

#include <deque>
#include <iostream>

using namespace Galois::Runtime::Distributed;

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

  

public:
  MPIBase() {
    int rc = MPI_Init (NULL, NULL);
    handleError(rc);
    
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskRank);

    // std::cout << "I am " << taskRank << " of " << numTasks << "\n";

    networkHostID = taskRank;
    networkHostNum = numTasks;

    if (taskRank == 0) {
      //master
    } else {
      //slave
    }
  }

  ~MPIBase() {
    MPI_Finalize();
  }

  //! sends a message.  assumes it is being called from a thread for which
  //! this is valid
  void sendInternal(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    buf.serialize(recv);
    int rv = MPI_Send(buf.linearData(), buf.size(), MPI_BYTE, dest, FuncTag, MPI_COMM_WORLD);
    handleError(rv);
  }

  void broadcastInternal(recvFuncTy recv, SendBuffer& buf) {
    buf.serialize(recv);
    for (int i = 0; i < numTasks; ++i) {
      if (i != taskRank) {
	int rv = MPI_Send(buf.linearData(), buf.size(), MPI_BYTE, i, FuncTag, MPI_COMM_WORLD);
	handleError(rv);
      }
    }
  }

  bool recvInternal() {
    int flag, rv;
    bool retval = false;
    MPI_Status status;
    do {
      //async probe
      rv = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
      handleError(rv);
      if (!flag)
	break;
      //use the lock
      lock.lock();
      rv = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
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
	buf.deserialize_end(f);
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
    bool bcast;
    uint32_t dest;
    recvFuncTy recv;
    SendBuffer buf;
    outgoingMessage(bool _bcast, uint32_t _dest, recvFuncTy _recv, SendBuffer& _buf)
      :bcast(_bcast), dest(_dest), recv(_recv), buf(std::move(_buf))
    {}

  };
  std::deque<outgoingMessage> asyncOutQueue;
  Galois::Runtime::LL::SimpleLock<true> asyncOutLock;

public:
  virtual ~NetworkInterfaceSyncMPI() {
  }

  virtual void sendMessage(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    asyncOutLock.lock();
    asyncOutQueue.push_back(outgoingMessage(false, dest, recv, buf));
    asyncOutLock.unlock();
  }

  virtual void broadcastMessage(recvFuncTy recv, SendBuffer& buf) {
    asyncOutLock.lock();
    asyncOutQueue.push_back(outgoingMessage(true, ~0, recv, buf));
    asyncOutLock.unlock();
  }

  virtual void systemBarrier() {
    int rv;
    rv = MPI_Barrier(MPI_COMM_WORLD);
    handleError(rv);
    return;
  }

  virtual bool handleReceives() {
    //assert master thread
    asyncOutLock.lock();
    while (!asyncOutQueue.empty()) {
      if (asyncOutQueue[0].bcast)
	broadcastInternal(asyncOutQueue[0].recv, asyncOutQueue[0].buf);
      else
	sendInternal(asyncOutQueue[0].dest, asyncOutQueue[0].recv, asyncOutQueue[0].buf);
      asyncOutQueue.pop_front();
    }
    asyncOutLock.unlock();

    return recvInternal();
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

  virtual void sendMessage(uint32_t dest, recvFuncTy recv, SendBuffer& buf) {
    sendInternal(dest, recv, buf);
  }

  virtual void broadcastMessage(recvFuncTy recv, SendBuffer& buf) {
    broadcastInternal(recv, buf);
  }

  virtual bool handleReceives() {
    return recvInternal();;
  }

  virtual bool needsDedicatedThread() {
    return false;
  }

  virtual void systemBarrier() {
    return;
  }

};

}


NetworkInterface& Galois::Runtime::Distributed::getSystemNetworkInterface() {
  static NetworkInterfaceSyncMPI net;
  return net;
}

void Galois::Runtime::Distributed::networkStart() {
  NetworkInterface& net = getSystemNetworkInterface();
  if (networkHostID != 0) {
    while (true) {
      net.handleReceives();
    }
  }
  return;
}

void Galois::Runtime::Distributed::networkTerminate() {
  if (networkHostNum > 1 && networkHostID == 0) {
    NetworkInterface& net = getSystemNetworkInterface();
    int x = 0;
    SendBuffer buf;
    buf.serialize(x);
    net.broadcastMessage (NULL, buf);
 // THIS SHOULD BE REMOVED IN CASE OF A DEDICATED COMM THREAD
    net.handleReceives();
    net.systemBarrier();
  }
  return;
}

