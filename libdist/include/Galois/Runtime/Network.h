/** Galois Network Layer -*- C++ -*-
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

#ifndef GALOIS_RUNTIME_NETWORK_H
#define GALOIS_RUNTIME_NETWORK_H

#include "Galois/Runtime/Serialize.h"
#include "Galois/Substrate/Barrier.h"

#include <cstdint>
/* Test for GCC >= 5.2.0 */
#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 1)
#include <experimental/tuple>
#include <experimental/optional>
#else
#include <utility>
#include <tuple>
#include <type_traits>
namespace { //anon
template <class F, class Tuple, size_t... I>
constexpr decltype(auto) apply_impl(  // exposition only
        F&& f, Tuple&& t, std::index_sequence<I...>) {
    return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}
template <class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
          std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}
} //end anon namespace
#include <boost/optional.hpp>
#endif

namespace galois {
namespace runtime {

typedef SerializeBuffer SendBuffer;
typedef DeSerializeBuffer RecvBuffer;

template<typename T>
/* Test for GCC >= 5.2.0 */
#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 1)
using optional_t = std::experimental::optional<T>;
#else
using optional_t = boost::optional<T>;
#endif

class NetworkInterface {

public:

  static uint32_t ID;
  static uint32_t Num;

  virtual ~NetworkInterface();

  //!send a message to a given (dest) host.  A message is simply a
  //!landing pad (recv, funciton pointer) and some data (buf)
  //! on the receiver, recv(buf) will be called durring handleReceives()
  //! buf is invalidated by this operation
  void sendMsg(uint32_t dest, void (*recv)(uint32_t, RecvBuffer&), SendBuffer& buf);

  //! send a message letting the network handle the serialization and deserialization
  //! slightly slower
  template<typename... Args>
  void sendSimple(uint32_t dest, void (*recv)(uint32_t, Args...), Args... param);

  //!send a message to a given (dest) host.  A message is simply a
  //!tag (tag) and some data (buf)
  //! on the receiver, buf will be returned on a receiveTagged(tag)
  //! buf is invalidated by this operation
  virtual void sendTagged(uint32_t dest, uint32_t tag, SendBuffer& buf) = 0;

  //!send a message to all hosts.  A message is simply a
  //!landing pad (recv) and some data (buf)
  //! buf is invalidated by this operation
  void broadcast(void (*recv)(uint32_t, RecvBuffer&), SendBuffer& buf, bool self = false);

  //! broadcast a mssage allowing the network to handle serialization and deserialization
  template<typename... Args>
  void broadcastSimple(void (*recv)(uint32_t, Args...), Args... param);

  //!receive and dispatch messages
  void handleReceives();

  //!receive tagged message
  virtual optional_t<std::pair<uint32_t, RecvBuffer>> recieveTagged(uint32_t tag, std::unique_lock<substrate::SimpleLock>* rlg) = 0;

  //!move send buffers out to network
  virtual void flush() = 0;

  virtual unsigned long reportSendBytes() const = 0;
  virtual unsigned long reportSendMsgs() const = 0;
  virtual unsigned long reportRecvBytes() const = 0;
  virtual unsigned long reportRecvMsgs() const = 0;
  virtual std::vector<unsigned long> reportExtra() const = 0;
  virtual std::vector<std::pair<std::string, unsigned long> > reportExtraNamed() const = 0;
};

extern uint32_t evilPhase;

NetworkInterface& getSystemNetworkInterface();
uint32_t getHostID();

NetworkInterface& makeNetworkBuffered();
NetworkInterface& makeNetworkRouted();

//! Host Barrier
substrate::Barrier& getHostBarrier();
////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////
namespace { //anon
template<typename... Args>
static void genericLandingPad(uint32_t src, RecvBuffer& buf) {
  void (*fp)(uint32_t, Args...);
  std::tuple<Args...> args;
  gDeserialize(buf, fp, args);
/* Test for GCC >= 5.2.0 */
#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 1)
  std::experimental::apply([fp, src](Args... params) { fp(src, params...); }, args);
#else
  apply([fp, src](Args... params) { fp(src, params...); }, args);
#endif
}
} //end anon namespace

template<typename... Args>
void NetworkInterface::sendSimple(uint32_t dest, void (*recv)(uint32_t, Args...), Args... param) {
  SendBuffer buf;
  gSerialize(buf, (uintptr_t)recv, param..., (uintptr_t)genericLandingPad<Args...>);
  sendTagged(dest, 0, buf);
}

template<typename... Args>
void NetworkInterface::broadcastSimple(void (*recv)(uint32_t, Args...), Args... param) {
  SendBuffer buf;
  gSerialize(buf, (uintptr_t)recv, param...);
  broadcast(genericLandingPad<Args...>, buf, false);
}

} //Runtime
} //Galois
#endif
