/** Reduction type -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Contains support for reducable objects
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_REDUCTION_H
#define GALOIS_REDUCTION_H

//FIXME: This duplicates and subsumes much of Accumulator.h

namespace Galois {

using Runtime::Distributed::networkHostID;
using Runtime::Distributed::networkHostNum;

template<typename T, typename BinFunc>
class DGReducible {
  BinFunc m_func;
  T m_initial;
  Runtime::PerThreadStorage<T> m_data;
  std::vector<DGReducible*> hosts;
  T r_data;

  volatile int reduced;

  void localReset(const T& init) {
    for (unsigned int i = 0; i < m_data.size(); ++i)
      *m_data.getRemote(i) = init;
    r_data = init;
  }

  static int expected() {
    int retval = 0;
    if (networkHostID * 2 + 1 < networkHostNum)
      ++retval;
    if (networkHostID * 2 + 2 < networkHostNum)
      ++retval;
    return retval;
  }

  void reduceWith(T& data) {
    r_data = m_func(r_data, data);
  }

  void localReduce() {
    for (unsigned int i = 0; i != m_data.size(); ++i)
      reduceWith(*m_data.getRemote(i));
  }

public:

  static void broadcastData(RecvBuffer& buf) {
    //std::cout << "B: " << networkHostID << "\n";
    DGReducible* dst;
    std::vector<DGReducible*> hosts;
    T data;
    gDeserialize(buf, hosts, data);
    dst = hosts[networkHostID];
    dst->localReset(data);
    dst->hosts = hosts;
  }

  static void registerInstance(RecvBuffer& buf) {
    assert(networkHostID == 0);
    DGReducible* dst;
    uint32_t host;
    DGReducible* ptr;
    gDeserialize(buf, dst, host, ptr);
    dst->hosts[host] = ptr;
  }

  static void reduceData(RecvBuffer& buf) {
    //std::cout << "R: " << networkHostID << "\n";
    DGReducible* dst;
    std::vector<DGReducible*> hosts;
    T data;
    gDeserialize(buf, hosts, data);
    dst = hosts[networkHostID];
    dst->hosts = hosts;
    dst->reduced++;
    dst->reduceWith(data);
    int expect = expected();
    if (expect == dst->reduced && networkHostID != 0) {
      //std::cout << "r: " << networkHostID << "->" << ((networkHostID - 1)/2) << "\n";
      dst->reduced = 0;
      SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->r_data);
      dst->r_data = dst->m_initial; //Reset reduce buffer
      getSystemNetworkInterface().sendMessage((networkHostID - 1)/2, &reduceData, sbuf);
    }
  }

  static void startReduce(RecvBuffer& buf) {
    //std::cout << "S: " << networkHostID << "\n";
    std::vector<DGReducible*> hosts;
    gDeserialize(buf, hosts);
    DGReducible* dst = hosts[networkHostID];
    dst->hosts = hosts;
    dst->localReduce();
    if (expected() == 0) {
      //std::cout << "s: " << networkHostID << "->" << ((networkHostID - 1)/2) << "\n";
      SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->r_data);
      getSystemNetworkInterface().sendMessage((networkHostID - 1)/2, &reduceData, sbuf);
    }
  }

  T& doReduce() {
    //Send reduce messages
    SendBuffer sbuf;
    gSerialize(sbuf, hosts);
    getSystemNetworkInterface().broadcastMessage(&startReduce, sbuf);

    int expect = expected();
    r_data = m_initial;
    localReduce();

    while (expect != reduced) {
      //spin processing network packets
      assert(Galois::Runtime::LL::getTID() == 0);
      getSystemNetworkInterface().handleReceives();
    }
    reduced = 0;
    return r_data;
  }

  void doBroadcast(T& data) {
    localReset(data);
    SendBuffer sbuf;
    gSerialize(sbuf, hosts, data);
    getSystemNetworkInterface().broadcastMessage(&broadcastData, sbuf, false);
  }

  T& get() {
    return *m_data.getLocal();
  }

  DGReducible(const BinFunc& f = BinFunc(), const T& initial = T()) :m_func(f), m_initial(initial), reduced(0) {
    hosts.resize(networkHostNum);
    hosts[networkHostID] = this;
    localReset(m_initial);
  }

  DGReducible(Galois::Runtime::Distributed::DeSerializeBuffer& s) :reduced(0) {
    gDeserialize(s, m_func, m_initial, hosts);
    localReset(m_initial);
    SendBuffer buf;
    gSerialize(buf, hosts[0], networkHostID, this);
    getSystemNetworkInterface().sendMessage(0, &registerInstance, buf);
  }

  // mark the graph as persistent so that it is distributed
  typedef int tt_is_persistent;
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    //This is what is called on the source of a replicating source
    gSerialize(s, m_func, m_initial, hosts);
  }
};



template<typename T, typename BinFunc>
class DGReducibleInplace {
  BinFunc m_func;
  T m_data;
  std::vector<DGReducibleInplace*> hosts;
  volatile int reduced;

  static int expected() {
    int retval = 0;
    if (networkHostID * 2 + 1 < networkHostNum)
      ++retval;
    if (networkHostID * 2 + 2 < networkHostNum)
      ++retval;
    return retval;
  }

public:

  static void broadcastData(RecvBuffer& buf) {
    //std::cout << "B: " << networkHostID << "\n";
    DGReducibleInplace* dst;
    std::vector<DGReducibleInplace*> hosts;
    T data;
    gDeserialize(buf, hosts);
    dst = hosts[networkHostID];
    gDeserialize(buf,dst->m_data);
    dst->doBroadcast(data);
  }

  static void registerInstance(RecvBuffer& buf) {
    assert(networkHostID == 0);
    DGReducibleInplace* dst;
    uint32_t host;
    DGReducibleInplace* ptr;
    gDeserialize(buf, dst, host, ptr);
    dst->hosts[host] = ptr;
  }

  static void reduceData(RecvBuffer& buf) {
    //std::cout << "R: " << networkHostID << "\n";
    DGReducibleInplace* dst;
    std::vector<DGReducibleInplace*> hosts;
    T data;
    gDeserialize(buf, hosts, data);
    dst = hosts[networkHostID];
    dst->hosts = hosts;
    dst->reduced++;
    dst->m_func(dst->m_data, data);
    int expect = expected();
    if (expect == dst->reduced && networkHostID != 0) {
      //std::cout << "r: " << networkHostID << "->" << ((networkHostID - 1)/2) << "\n";
      dst->reduced = 0;
      SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->r_data);
      getSystemNetworkInterface().sendMessage((networkHostID - 1)/2, &reduceData, sbuf);
    }
  }

  static void startReduce(RecvBuffer& buf) {
    //std::cout << "S: " << networkHostID << "\n";
    std::vector<DGReducibleInplace*> hosts;
    gDeserialize(buf, hosts);
    DGReducibleInplace* dst = hosts[networkHostID];
    dst->hosts = hosts;
    if (expected() == 0) {
      //std::cout << "s: " << networkHostID << "->" << ((networkHostID - 1)/2) << "\n";
      SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->m_data);
      getSystemNetworkInterface().sendMessage((networkHostID - 1)/2, &reduceData, sbuf);
    }
  }

  T& doReduce() {
    //Send reduce messages
    SendBuffer sbuf;
    gSerialize(sbuf, hosts);
    getSystemNetworkInterface().broadcastMessage(&startReduce, sbuf);

    int expect = expected();

    while (expect != reduced) {
      //spin processing network packets
      assert(Galois::Runtime::LL::getTID() == 0);
      getSystemNetworkInterface().handleReceives();
    }
    reduced = 0;
    return m_data;
  }

  T& get() {
    return m_data;
  }

  DGReducibleInplace(const BinFunc& f = BinFunc()) :m_func(f), reduced(0) {
    hosts.resize(networkHostNum);
    hosts[networkHostID] = this;
  }

  DGReducibleInplace(Galois::Runtime::Distributed::DeSerializeBuffer& s) :reduced(0) {
    gDeserialize(s, m_func, hosts);
    SendBuffer buf;
    gSerialize(buf, hosts[0], networkHostID, this);
    getSystemNetworkInterface().sendMessage(0, &registerInstance, buf);
  }

  // mark the graph as persistent so that it is distributed
  typedef int tt_is_persistent;
  typedef int tt_has_serialize;
  void serialize(Galois::Runtime::Distributed::SerializeBuffer& s) const {
    //This is what is called on the source of a replicating source
    gSerialize(s, m_func, hosts);
  }
};


} //namespace Galois

#endif
