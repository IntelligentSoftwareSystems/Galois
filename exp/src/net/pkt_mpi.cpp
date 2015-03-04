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

#include "Galois/Runtime/NetworkIO.h"

#include <mpi.h>
#include <deque>
//#include <iostream>

class NetworkIOMPI : public Galois::Runtime::NetworkIO {

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


  struct sendQueueTy {
    struct mpiMessage {
      std::vector<uint8_t> data;
      MPI_Request req;
      mpiMessage(std::vector<uint8_t>& m) { data.swap(m); }
    };

    uint32_t myDest;
    std::deque<mpiMessage> inflight;

    void complete() {
      while (!inflight.empty()) {
        int flag = 0;
        MPI_Status status;
        auto& f = inflight.front();
        MPI_Test(&f.req, &flag, &status);
        if (flag) {
          inflight.pop_front();
          //          std::cerr << "S";
        } else {
          break;
        }
      }
    }

    void send(std::vector<uint8_t>& m) {
      inflight.push_back({m});
      auto& mb = inflight.back();
      int rv = MPI_Isend(mb.data.data(), mb.data.size(), MPI_BYTE, myDest, 0, MPI_COMM_WORLD, &mb.req);
      //      std::cerr << "s";
      handleError(rv);
    }
  };

  struct recvQueueTy {
    struct mpiMessage {
      std::vector<uint8_t> data;
      MPI_Request req;
      mpiMessage(uint32_t n) :data(n) {}
    };

    std::deque<mpiMessage> inflight;
    std::deque<std::vector<uint8_t> > done;

    void probe() {
      int flag = 0;
      do {
        MPI_Status status;
        int rv = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        handleError(rv);
        if (flag) {
          int nbytes;
          rv = MPI_Get_count(&status, MPI_BYTE, &nbytes);
          handleError(rv);
          inflight.push_back(mpiMessage{static_cast<uint32_t>(nbytes)});
          auto& r = inflight.back();
          rv = MPI_Irecv(r.data.data(), nbytes, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &r.req);
          //          std::cerr << "r";
          handleError(rv);
        }
      } while(flag);
    }

    void complete() {
      while (!inflight.empty()) {
        int flag;
        MPI_Status status;
        auto& r = inflight.front();
        int rv = MPI_Test(&r.req, &flag, &status);
        handleError(rv);
        if (!flag)
          break;
        int nbytes;
        rv = MPI_Get_count(&status, MPI_BYTE, &nbytes);
        handleError(rv);
        //        std::cerr << "R";
        done.emplace_back(std::move(r.data));
        inflight.pop_front();
      }
    }
  };

  std::vector<sendQueueTy> sendQueues;
  recvQueueTy recvQueue;


  void progress() {
    for(auto& q : sendQueues)
      q.complete();
    recvQueue.complete();
    recvQueue.probe();
  }

public:

  NetworkIOMPI(uint32_t& ID, uint32_t& NUM) {
    auto p = initMPI();
    ID = p.first;
    NUM = p.second;
    sendQueues.resize(NUM);
    for (uint32_t i = 0; i < NUM; ++i) {
      sendQueues[i].myDest = i;
    }
  }

  ~NetworkIOMPI() {
    int rv = MPI_Finalize();
    handleError(rv);
  }
  
  virtual void enqueue(uint32_t dest, std::vector<uint8_t>& data) {
    progress();
    auto& sq = sendQueues[dest];
    sq.send(data);
  }

  virtual std::vector<uint8_t> dequeue() {
    progress();
    std::vector<uint8_t> retval;
    if (!recvQueue.done.empty()) {
      retval.swap(recvQueue.done.front());
      recvQueue.done.pop_front();
    }
    return retval;
  }

};

std::tuple<Galois::Runtime::NetworkIO*,uint32_t,uint32_t> Galois::Runtime::makeNetworkIOMPI() {
  uint32_t ID, NUM;
  Galois::Runtime::NetworkIO* n = new NetworkIOMPI(ID, NUM);
  return std::make_tuple(n, ID, NUM);
}

