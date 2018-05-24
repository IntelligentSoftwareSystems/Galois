#ifndef GALOIS_RUNTIME_NETWORKTHREAD_H
#define GALOIS_RUNTIME_NETWORKTHREAD_H

#include <cstdint>
#include <vector>
#include <tuple>
#include <memory>
#include <cassert>
#include <cstring>
#include <deque>
#include <string>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <mpi.h>
#include "galois/runtime/MemUsage.h"

namespace galois {
namespace runtime {
class NetworkIO {
protected:
  /**
   * Wrapper for dealing with MPI error codes. Dies if the error code isn't
   * MPI_SUCCESS.
   *
   * @param rc Error code to check for success
   */
  static void handleError(int rc) {
    if (rc != MPI_SUCCESS) {
      MPI_Abort(MPI_COMM_WORLD, rc);
    }
  }

  MemUsageTracker& memUsageTracker;
public:
  struct message {
    uint32_t host;
    uint32_t tag;
    std::vector<uint8_t> data;

    message() :host(~0), tag(~0) {}
    message(uint32_t h, uint32_t t, std::vector<uint8_t>&& d) 
      : host(h), tag(t), data(d) {}

    bool valid() const { return !data.empty(); }
  };

  NetworkIO(MemUsageTracker& tracker): memUsageTracker(tracker) {}

  virtual ~NetworkIO();
  
  // Takes ownership of data buffer
  virtual void enqueue(message m) = 0;
  // Returns empty if no message
  virtual message dequeue() = 0;
  // Make progress.  other functions don't have to
  virtual void progress() = 0;

  //void operator() () -- make progress
  //bool readySend() -- can send
  //bool readyRecv() -- packet waiting
  //void send(const message&) -- send data
  //message recv() -- receive data
};

std::tuple<std::unique_ptr<NetworkIO>, uint32_t, uint32_t> makeNetworkIOMPI(galois::runtime::MemUsageTracker& tracker);
#ifdef GALOIS_USE_LWCI
std::tuple<std::unique_ptr<NetworkIO>, uint32_t, uint32_t> makeNetworkIOLWCI(galois::runtime::MemUsageTracker& tracker);
#endif

} //namespace runtime
} //namespace galois

#endif
