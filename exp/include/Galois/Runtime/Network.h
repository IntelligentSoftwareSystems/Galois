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
 * @author Manoj Dhanapal <madhanap@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_NETWORK_H
#define GALOIS_RUNTIME_NETWORK_H

#include "Galois/Runtime/Serialize.h"

#include <cstdint>
#include <tuple>
#include <unordered_map>

//#define USE_TCP
#define USE_MPI
//#define USE_BUF

namespace Galois {
namespace Runtime {

typedef SerializeBuffer SendBuffer;
typedef DeSerializeBuffer RecvBuffer;

typedef void (*recvFuncTy)(RecvBuffer&);

class NetworkInterface {
public:

  static uint32_t ID;
  static uint32_t Num;

  virtual ~NetworkInterface();

  //!send a message to a given (dest) host.  A message is simply a
  //!landing pad (recv) and some data (buf)
  //! buf is invalidated by this operation
  virtual void send(uint32_t dest, recvFuncTy recv, SendBuffer& buf) = 0;

  //!send a message to all hosts.  A message is simply a
  //!landing pad (recv) and some data (buf)
  //! buf is invalidated by this operation
  void broadcast(recvFuncTy recv, SendBuffer& buf, bool self = false);

  //! send a message letting the network handle the serialization and deserialization
  //! slightly slower
  template<typename... Args>
  void sendAlt(uint32_t dest, void (*recv)(Args...), Args... param);

  //! broadcast a mssage allowing the network to handle serialization and deserialization
  template<typename... Args>
  void broadcastAlt(void (*recv)(Args...), Args... param);

  //!receive and dispatch messages
  //!returns true if at least one message was received
  //! if the network requires a dedicated thread, then 
  //!it is only valid for that thread to call this function
  virtual bool handleReceives() = 0;

  virtual void flush();

  //! start a listen loop if not the lead process
  //! FIXME: should this be split out?
  static void start();

  //! terminate all processes
  //! FIXME: should this be split out?
  static void terminate();

  //! send a top level loop item (executed in the top level event loop)
  //! FIXME: Why does this exist?
  static void sendLoop(uint32_t dest, recvFuncTy recv, SendBuffer& buf);

};

NetworkInterface& getSystemNetworkInterface();
uint32_t getHostID();

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////
template<typename... Args>
struct genericLandingPad {
  template<int ...> struct seq {};
  template<int N, int ...S> struct gens : gens<N-1, N-1, S...> {};
  template<int ...S> struct gens<0, S...>{ typedef seq<S...> type; };
  
  template<int ...S>
  static void callFunc(void (*fp)(Args...), std::tuple<Args...>& args, seq<S...>)
  {
    return fp(std::move(std::get<S>(args)) ...);
  }

  //do this the new fancy way
  static void func(RecvBuffer& buf) {
    //    std::cerr << "RA\n";
    void (*fp)(Args...);
    std::tuple<Args...> args;
    gDeserialize(buf, fp, args);
    //    std::cerr << "end RA " << fp << "\n";
    callFunc(fp, args, typename gens<sizeof...(Args)>::type());
  }
};

template<typename... Args>
void NetworkInterface::sendAlt(uint32_t dest, void (*recv)(Args...), Args... param) {
  //  std::cerr << "SA : " << dest << " " << (void*)recv << " " << sizeof...(Args) << "\n";
  SendBuffer buf;
  gSerialize(buf, recv, param...);
  //  std::cerr << "end SA\n";
  send(dest, genericLandingPad<Args...>::func, buf);
  //  std::cerr << "end SA\n";
}

template<typename... Args>
void NetworkInterface::broadcastAlt(void (*recv)(Args...), Args... param) {
  SendBuffer buf;
  gSerialize(buf, recv, param...);
  broadcast(genericLandingPad<Args...>::func, buf, false);
}

} //Runtime
} //Galois
#endif
