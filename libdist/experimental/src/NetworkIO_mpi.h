/** Galois Network Backend for MPI -*- C++ -*-
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

#include "galois/runtime/NetworkIO.h"

#include <mpi.h>
#include <deque>
#include <atomic>
#include <cassert>
#include <iostream>

namespace {

static void handleError(int rc) {
  if (rc != MPI_SUCCESS) {
    //GALOIS_ERROR(false, "MPI ERROR"); 
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
}

std::pair<int, int> initMPI() {
  int provided;
  int rc = MPI_Init_thread (NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  handleError(rc);
  
  int numTasks, taskRank;
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskRank);
  
  return std::make_pair(taskRank, numTasks);
}


class NetworkIOMPI : public galois::runtime::NetworkIO {
public:
  using NetworkIO::message;
  
private:

  struct mpi_message {
    bool inflight;
    MPI_Request cur;
    message m;
  };

  std::deque<message> inq;
  std::vector<mpi_message> outq;
  std::atomic<int> incount;
  int _ID, _Num;

  void printBuffer(bool recv, int me, int from, std::vector<char>& d) {
    std::ostringstream s;
    s << me << (recv ? ": R " : ": S ") << from << " D " << d.size();
    // for (char c : d)
    //   s << " " << (int) c;
    s << "\n";
    std::cerr << s.str();
  }

public:
  
  NetworkIOMPI() :incount(0) {
    auto p = initMPI();
    _ID = p.first;
    _Num = p.second;
    outq.resize(_Num);
  }
  ~NetworkIOMPI() {
   // MPI_Finalize();
  }

  void operator()() {
    int rv;
    
    //Check for recv
    MPI_Status status;
    int nbytes, flag;
    rv = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    handleError(rv);
    if (flag) { //pending message
      rv = MPI_Get_count(&status, MPI_CHAR, &nbytes);
      handleError(rv);
      assert(nbytes != MPI_UNDEFINED);
      std::vector<char> d(nbytes);
      std::cerr << _ID << " mpi_recv " << nbytes << " " << status.MPI_SOURCE << "\n";
      rv = MPI_Recv(d.data(), nbytes, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
      //printBuffer(true, _ID, status.MPI_SOURCE, d);
      handleError(rv);
      bool urg = status.MPI_TAG;
      inq.push_back(message{static_cast<uint32_t>(status.MPI_SOURCE), std::move(d), urg});
      ++incount;
    }
    
    for (int i = 0; i < _Num; ++i) {
      auto& q = outq[i];
      if (q.inflight) {
        //try completing
        MPI_Test(&q.cur, &flag, &status);
        if (flag)
          q.inflight = false;
      }
    }
  }

  bool readySend() const {
    return true;
  }

  bool readySend(uint32_t dest) const {
    return !outq[dest].inflight;
  }

  bool readyRecv() const {
    return incount;
  }

  void send(const message& m) {
    auto& q = outq[m.dest];
    assert(!q.inflight);
    q.inflight = true;
    q.m = std::move(m);
    //printBuffer(false, _ID, m.dest, q.m.data);
    int rv = MPI_Isend(q.m.data.data(), q.m.data.size(), MPI_BYTE, q.m.dest, q.m.urgent, MPI_COMM_WORLD, &q.cur);
    handleError(rv);
  }

  message recv() {
    assert(incount);
    message retval = std::move(inq.front());
    inq.pop_front();
    --incount;
    return retval;
  }

  uint32_t ID() const { return _ID; }
  uint32_t Num() const { return _Num; }

};

}
