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
 * @file Network.h
 *
 * Contains the network interface class which is the base class for all
 * network layer implementations.
 */

#ifndef GALOIS_RUNTIME_NETWORK_H
#define GALOIS_RUNTIME_NETWORK_H

#include "galois/runtime/Serialize.h"
#include "galois/runtime/MemUsage.h"
#include "galois/substrate/Barrier.h"

#include <cstdint>
/* Test for GCC >= 5.2.0 */
#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 1)
#include <experimental/tuple>
#include <experimental/optional>
#else
#include <utility>
#include <tuple>
#include <type_traits>

namespace { // anon
template <class F, class Tuple, size_t... I>
constexpr decltype(auto) apply_impl( // exposition only
    F&& f, Tuple&& t, std::index_sequence<I...>) {
  return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}
template <class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return apply_impl(
      std::forward<F>(f), std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}
} // namespace
#include <boost/optional.hpp>
#endif

#include <mpi.h>

namespace galois {
namespace runtime {

//! typedef for buffer that stores data to be sent out
using SendBuffer = SerializeBuffer;
//! typedef for buffer that received data is saved into
using RecvBuffer = DeSerializeBuffer;

//! Optional type wrapper
template <typename T>
/* Test for GCC >= 5.2.0 */
#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 1)
using optional_t = std::experimental::optional<T>;
#else
using optional_t = boost::optional<T>;
#endif

/**
 * A class that defines functions that a network interface in Galois should
 * have. How the sends/recvs/stat-collecting happens as well
 * as the network layer itself is up to the implemention of the class.
 */
class NetworkInterface {
protected:
  //! Initialize the MPI system. Should only be called once per process.
  void initializeMPI();

  //! Memory usage tracker
  MemUsageTracker memUsageTracker;

#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
public:
  //! Wrapper that calls into increment mem usage on the memory usage tracker
  inline void incrementMemUsage(uint64_t size) {
    memUsageTracker.incrementMemUsage(size);
  }
  //! Wrapper that calls into decrement mem usage on the memory usage tracker
  inline void decrementMemUsage(uint64_t size) {
    memUsageTracker.decrementMemUsage(size);
  }
#endif

public:
  //! This machine's host ID
  static uint32_t ID;
  //! The total number of machines in the current program
  static uint32_t Num;

  /**
   * Constructor for interface.
   */
  NetworkInterface();

  /**
   * Destructor destroys MPI (if it exists).
   */
  virtual ~NetworkInterface();

  //! Send a message to a given (dest) host.  A message is simply a
  //! landing pad (recv, funciton pointer) and some data (buf)
  //! on the receiver, recv(buf) will be called durring handleReceives()
  //! buf is invalidated by this operation
  void sendMsg(uint32_t dest, void (*recv)(uint32_t, RecvBuffer&),
               SendBuffer& buf);

  //! Send a message letting the network handle the serialization and
  //! deserialization slightly slower
  template <typename... Args>
  void sendSimple(uint32_t dest, void (*recv)(uint32_t, Args...),
                  Args... param);

  //! Send a message to a given (dest) host.  A message is simply a
  //! tag (tag) and some data (buf)
  //! on the receiver, buf will be returned on a receiveTagged(tag)
  //! buf is invalidated by this operation
  virtual void sendTagged(uint32_t dest, uint32_t tag, SendBuffer& buf) = 0;

  //! Send a message to all hosts.  A message is simply a
  //! landing pad (recv) and some data (buf)
  //! buf is invalidated by this operation
  void broadcast(void (*recv)(uint32_t, RecvBuffer&), SendBuffer& buf,
                 bool self = false);

  //! Broadcast a message allowing the network to handle serialization and
  //! deserialization
  template <typename... Args>
  void broadcastSimple(void (*recv)(uint32_t, Args...), Args... param);

  //! Receive and dispatch messages
  void handleReceives();

  //! Wrapper to reset the mem usage tracker's stats
  inline void resetMemUsage() { memUsageTracker.resetMemUsage(); }

  //! Reports the memory usage tracker's statistics to the stat manager
  void reportMemUsage() const;

  //! Receive a tagged message
  virtual optional_t<std::pair<uint32_t, RecvBuffer>>
  recieveTagged(uint32_t tag, std::unique_lock<substrate::SimpleLock>* rlg) = 0;

  //! move send buffers out to network
  virtual void flush() = 0;

  //! @returns true if any send is in progress or is pending to be enqueued
  virtual bool anyPendingSends() = 0;

  //! @returns true if any receive is in progress or is pending to be dequeued
  virtual bool anyPendingReceives() = 0;

  //! Get how many bytes were sent
  //! @returns num bytes sent
  virtual unsigned long reportSendBytes() const = 0;
  //! Get how many messages were sent
  //! @returns num messages sent
  virtual unsigned long reportSendMsgs() const = 0;
  //! Get how many bytes were received
  //! @returns num bytes received
  virtual unsigned long reportRecvBytes() const = 0;
  //! Get how many messages were received
  //! @returns num messages received
  virtual unsigned long reportRecvMsgs() const = 0;
  //! Get any other extra statistics that might need to be reported; varies
  //! depending on implementation
  //! @returns vector of extra things to be reported
  virtual std::vector<unsigned long> reportExtra() const = 0;
  //! Get the names of the extra things that are returned by reportExtra
  //! @returns vector of the names of the reported extra things
  virtual std::vector<std::pair<std::string, unsigned long>>
  reportExtraNamed() const = 0;
};

//! Variable that keeps track of which network send/recv phase a program is
//! currently on. Can be seen as a count of send/recv rounds that have occured.
extern uint32_t evilPhase;

//! Get the network interface
//! @returns network interface
NetworkInterface& getSystemNetworkInterface();
//! Gets this host's ID
//! @returns ID of this host
uint32_t getHostID();

//! Returns a BufferedNetwork interface
NetworkInterface& makeNetworkBuffered();

//! Returns a host barrier, which is a regular MPI-Like Barrier for all hosts.
//! @warning Should not be called within a parallel region; assumes only one
//! thread is calling it
substrate::Barrier& getHostBarrier();
//! Returns a fence that ensures all pending messages are delivered, acting
//! like a memory-barrier
substrate::Barrier& getHostFence();

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////
namespace { // anon
template <typename... Args>

static void genericLandingPad(uint32_t src, RecvBuffer& buf) {
  void (*fp)(uint32_t, Args...);
  std::tuple<Args...> args;
  gDeserialize(buf, fp, args);
/* Test for GCC >= 5.2.0 */
#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 1)
  std::experimental::apply([fp, src](Args... params) { fp(src, params...); },
                           args);
#else
  apply([fp, src](Args... params) { fp(src, params...); }, args);
#endif
}

} // namespace

template <typename... Args>
void NetworkInterface::sendSimple(uint32_t dest,
                                  void (*recv)(uint32_t, Args...),
                                  Args... param) {
  SendBuffer buf;
  gSerialize(buf, (uintptr_t)recv, param...,
             (uintptr_t)genericLandingPad<Args...>);
  sendTagged(dest, 0, buf);
}

template <typename... Args>
void NetworkInterface::broadcastSimple(void (*recv)(uint32_t, Args...),
                                       Args... param) {
  SendBuffer buf;
  gSerialize(buf, (uintptr_t)recv, param...);
  broadcast(genericLandingPad<Args...>, buf, false);
}

} // namespace runtime
} // namespace galois
#endif
