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
 * @file NetworkIO.h
 *
 * Contains NetworkIO, a base class that is inherited by classes that want to
 * implement the communication layer of Galois. (e.g. NetworkIOMPI and
 * NetworkIOLWCI)
 */

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

/**
 * Class for the network IO layer which is responsible for doing sends/receives
 * of data. Used by the network interface to do the actual communication.
 */
class NetworkIO {
protected:
  /**
   * Wrapper for dealing with MPI error codes. Program dies if the error code
   * isn't MPI_SUCCESS.
   *
   * @param rc Error code to check for success
   */
  static void handleError(int rc) {
    if (rc != MPI_SUCCESS) {
      MPI_Abort(MPI_COMM_WORLD, rc);
    }
  }

  //! memory usage tracker
  MemUsageTracker& memUsageTracker;

  //! Number of inflight sends and receives
  std::atomic<size_t>& inflightSends;
  std::atomic<size_t>& inflightRecvs;

public:
  /**
   * Message structure for sending data across the network.
   */
  struct message {
    uint32_t host; //!< destination of this message
    uint32_t tag;  //!< tag on message indicating distinct communication phases
    std::vector<uint8_t> data; //!< data portion of message

    //! Default constructor initializes host and tag to large numbers.
    message() : host(~0), tag(~0) {}
    //! @param h Host to send message to
    //! @param t Tag to associate with message
    //! @param d Data to save in message
    message(uint32_t h, uint32_t t, std::vector<uint8_t>&& d)
        : host(h), tag(t), data(d) {}

    //! A message is valid if there is data to be sent
    //! @returns true if data is non-empty
    bool valid() const { return !data.empty(); }
  };

  //! The default constructor takes a memory usage tracker and saves it
  //! @param tracker reference to a memory usage tracker used by the system
  NetworkIO(MemUsageTracker& tracker, std::atomic<size_t>& sends, std::atomic<size_t>& recvs)
    : memUsageTracker(tracker), inflightSends(sends), inflightRecvs(recvs) {}

  //! Default destructor does nothing.
  virtual ~NetworkIO();
  //! Queues a message for sending out. Takes ownership of data buffer.
  virtual void enqueue(message m) = 0;
  //! Checks to see if a message is here for this host to receive. If so, take
  //! and return it
  //! @returns an empty message if no message
  virtual message dequeue() = 0;
  //! Make progress. Other functions don't have to make progress.
  virtual void progress() = 0;
};

/**
 * Creates/returns a network IO layer that uses MPI to do communication.
 *
 * @returns tuple with pointer to the MPI IO layer, this host's ID, and the
 * total number of hosts in the system
 */
std::tuple<std::unique_ptr<NetworkIO>, uint32_t, uint32_t>
makeNetworkIOMPI(galois::runtime::MemUsageTracker& tracker, std::atomic<size_t>& sends, std::atomic<size_t>& recvs);
#ifdef GALOIS_USE_LWCI
/**
 * Creates/returns a network IO layer that uses LWCI to do communication.
 *
 * @returns tuple with pointer to the LWCI IO layer, this host's ID, and the
 * total number of hosts in the system
 */
std::tuple<std::unique_ptr<NetworkIO>, uint32_t, uint32_t>
makeNetworkIOLWCI(galois::runtime::MemUsageTracker& tracker, std::atomic<size_t>& sends, std::atomic<size_t>& recvs);
#endif

} // namespace runtime
} // namespace galois

#endif
