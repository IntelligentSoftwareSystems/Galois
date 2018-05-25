/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_OLD_DIST_REDUCTION_H
#define GALOIS_OLD_DIST_REDUCTION_H

//FIXME: This duplicates and subsumes much of Accumulator.h
//FIXME: This is complete bogusness


#include "galois/runtime/PerHostStorage.h"

namespace galois {

template<typename T, typename BinFunc>
class DGReducible : public runtime::Lockable {
  BinFunc m_func;
  T m_initial;
  runtime::PerThreadStorage<T> m_data;
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
    if (galois::runtime::NetworkInterface::ID * 2 + 1 < galois::runtime::NetworkInterface::Num)
      ++retval;
    if (galois::runtime::NetworkInterface::ID * 2 + 2 < galois::runtime::NetworkInterface::Num)
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
  static void broadcastData(runtime::RecvBuffer& buf) {
    //std::cout << "B: " << galois::runtime::NetworkInterface::ID << "\n";
    DGReducible* dst;
    std::vector<DGReducible*> hosts;
    T data;
    gDeserialize(buf, hosts, data);
    dst = hosts[galois::runtime::NetworkInterface::ID];
    assert(dst);
    dst->localReset(data);
    dst->hosts = hosts;
  }

  static void registerInstance(runtime::RecvBuffer& buf) {
    assert(galois::runtime::NetworkInterface::ID == 0);
    DGReducible* dst;
    uint32_t host;
    DGReducible* ptr;
    gDeserialize(buf, dst, host, ptr);
    dst->hosts[host] = ptr;
  }

  static void reduceData(runtime::RecvBuffer& buf) {
    //std::cout << "R: " << galois::runtime::NetworkInterface::ID << "\n";
    DGReducible* dst;
    std::vector<DGReducible*> hosts;
    T data;
    gDeserialize(buf, hosts, data);
    dst = hosts[galois::runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    dst->reduced++;
    dst->reduceWith(data);
    int expect = expected();
    if (expect == dst->reduced && galois::runtime::NetworkInterface::ID != 0) {
      //std::cout << "r: " << galois::runtime::NetworkInterface::ID << "->" << ((galois::runtime::NetworkInterface::ID - 1)/2) << "\n";
      dst->reduced = 0;
      runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->r_data);
      dst->r_data = dst->m_initial; //Reset reduce buffer
      runtime::getSystemNetworkInterface().send((galois::runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  static void startReduce(runtime::RecvBuffer& buf) {
    //std::cout << "S: " << galois::runtime::NetworkInterface::ID << "\n";
    std::vector<DGReducible*> hosts;
    gDeserialize(buf, hosts);
    DGReducible* dst = hosts[galois::runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    dst->localReduce();
    if (expected() == 0) {
      //std::cout << "s: " << galois::runtime::NetworkInterface::ID << "->" << ((galois::runtime::NetworkInterface::ID - 1)/2) << "\n";
      runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->r_data);
      runtime::getSystemNetworkInterface().send((galois::runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  T& doReduce() {
    //Send reduce messages
    runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts);
    runtime::getSystemNetworkInterface().broadcast(&startReduce, sbuf);

    int expect = expected();
    r_data = m_initial;
    localReduce();

    while (expect != reduced) {
      //spin processing network packets
      assert(runtime::LL::getTID() == 0);
      runtime::getSystemNetworkInterface().handleReceives();
    }
    reduced = 0;
    return r_data;
  }

  void doBroadcast(const T& data) {
    localReset(data);
    runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts, data);
    runtime::getSystemNetworkInterface().broadcast(&broadcastData, sbuf, false);
  }

  T& get() {
    return *m_data.getLocal();
  }

  DGReducible(const BinFunc& f = BinFunc(), const T& initial = T()) :m_func(f), m_initial(initial), reduced(0) {
    hosts.resize(galois::runtime::NetworkInterface::Num);
    hosts[galois::runtime::NetworkInterface::ID] = this;
    localReset(m_initial);
  }

  DGReducible(galois::runtime::DeSerializeBuffer& s) :reduced(0) {
    gDeserialize(s, m_func, m_initial, hosts);
    localReset(m_initial);
    runtime::SendBuffer buf;
    gSerialize(buf, hosts[0], galois::runtime::NetworkInterface::ID, this);
    runtime::getSystemNetworkInterface().send(0, &registerInstance, buf);
  }

  // mark the graph as persistent so that it is distributed
  typedef int tt_is_persistent;
  typedef int tt_has_serialize;
  void serialize(galois::runtime::SerializeBuffer& s) const {
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
    if (galois::runtime::NetworkInterface::ID * 2 + 1 < galois::runtime::NetworkInterface::Num)
      ++retval;
    if (galois::runtime::NetworkInterface::ID * 2 + 2 < galois::runtime::NetworkInterface::Num)
      ++retval;
    return retval;
  }

public:
  static void broadcastData(runtime::RecvBuffer& buf) {
    //std::cout << "B: " << galois::runtime::NetworkInterface::ID << "\n";
    DGReducibleInplace* dst;
    std::vector<DGReducibleInplace*> hosts;
    T data;
    gDeserialize(buf, hosts);
    dst = hosts[galois::runtime::NetworkInterface::ID];
    gDeserialize(buf, dst->m_data);
    dst->doBroadcast(data);
  }

  static void registerInstance(runtime::RecvBuffer& buf) {
    assert(galois::runtime::NetworkInterface::ID == 0);
    DGReducibleInplace* dst;
    uint32_t host;
    DGReducibleInplace* ptr;
    gDeserialize(buf, dst, host, ptr);
    dst->hosts[host] = ptr;
  }

  static void reduceData(runtime::RecvBuffer& buf) {
    //std::cout << "R: " << galois::runtime::NetworkInterface::ID << "\n";
    DGReducibleInplace* dst;
    std::vector<DGReducibleInplace*> hosts;
    T data;
    gDeserialize(buf, hosts, data);
    dst = hosts[galois::runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    dst->reduced++;
    dst->m_func(dst->m_data, data);
    int expect = expected();
    if (expect == dst->reduced && galois::runtime::NetworkInterface::ID != 0) {
      //std::cout << "r: " << galois::runtime::NetworkInterface::ID << "->" << ((galois::runtime::NetworkInterface::ID - 1)/2) << "\n";
      dst->reduced = 0;
      runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->m_data);
      runtime::getSystemNetworkInterface().send((galois::runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  static void startReduce(runtime::RecvBuffer& buf) {
    //std::cout << "S: " << galois::runtime::NetworkInterface::ID << "\n";
    std::vector<DGReducibleInplace*> hosts;
    gDeserialize(buf, hosts);
    DGReducibleInplace* dst = hosts[galois::runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    if (expected() == 0) {
      //std::cout << "s: " << galois::runtime::NetworkInterface::ID << "->" << ((galois::runtime::NetworkInterface::ID - 1)/2) << "\n";
      runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->m_data);
      runtime::getSystemNetworkInterface().send((galois::runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  T& doReduce() {
    //Send reduce messages
    runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts);
    runtime::getSystemNetworkInterface().broadcast(&startReduce, sbuf);

    int expect = expected();

    while (expect != reduced) {
      //spin processing network packets
      assert(runtime::LL::getTID() == 0);
      runtime::getSystemNetworkInterface().handleReceives();
    }
    reduced = 0;
    return m_data;
  }

  T& get() {
    return m_data;
  }

  DGReducibleInplace(const BinFunc& f = BinFunc()) :m_func(f), reduced(0) {
    hosts.resize(galois::runtime::NetworkInterface::Num);
    hosts[galois::runtime::NetworkInterface::ID] = this;
  }

  DGReducibleInplace(galois::runtime::DeSerializeBuffer& s) :reduced(0) {
    gDeserialize(s, m_func, hosts);
    runtime::SendBuffer buf;
    gSerialize(buf, hosts[0], galois::runtime::NetworkInterface::ID, this);
    runtime::getSystemNetworkInterface().send(0, &registerInstance, buf);
  }

  // mark the graph as persistent so that it is distributed
  typedef int tt_is_persistent;
  typedef int tt_has_serialize;
  void serialize(galois::runtime::SerializeBuffer& s) const {
    //This is what is called on the source of a replicating source
    gSerialize(s, m_func, hosts);
  }
};

template<typename T, typename BinFunc>
class DGReducibleVector {
  struct Item {
    int first;
    T second;

    Item(): first(0) { }
    Item(const T& t): first(0), second(t) { }

    Item(galois::runtime::DeSerializeBuffer& s) {
      galois::runtime::gDeserialize(s, second);
      first = 0;
    }

    typedef int tt_has_serialize;
    void serialize(galois::runtime::SerializeBuffer& s) const {
      galois::runtime::gSerialize(s, second);
    }

    void deserialize(galois::runtime::DeSerializeBuffer& s) {
      galois::runtime::gDeserialize(s, second);
      first = 0;
    }
  };

  typedef std::vector<Item> PerPackage;
  typedef runtime::PerPackageStorage<PerPackage> Data;

  BinFunc m_func;
  T m_initial;
  Data m_data;
  runtime::LL::SimpleLock lock;

  std::vector<DGReducibleVector*> hosts;
  size_t m_size;

  volatile int reduced;

  void localUpdate(const PerPackage& init) {
    for (unsigned int i = 0; i < m_data.size(); ++i) {
      if (!runtime::LL::isPackageLeader(i))
        continue;
      PerPackage& p = *m_data.getRemote(i);
      if (p.empty())
        continue;
      // TODO parallelize
      for (size_t x = 0; x < m_size; ++x)
        p[x] = init[x];
    }
  }

  void localUpdate(const T& init) {
    for (unsigned int i = 0; i < m_data.size(); ++i) {
      if (!runtime::LL::isPackageLeader(i))
        continue;
      PerPackage& p = *m_data.getRemote(i);
      if (p.empty())
        continue;
      // TODO parallelize
      for (size_t x = 0; x < m_size; ++x)
        p[x].second = init;
    }
  }

  void localUpdate() {
    localUpdate(m_initial);
  }

  static int expected() {
    int retval = 0;
    if (galois::runtime::NetworkInterface::ID * 2 + 1 < galois::runtime::NetworkInterface::Num)
      ++retval;
    if (galois::runtime::NetworkInterface::ID * 2 + 2 < galois::runtime::NetworkInterface::Num)
      ++retval;
    return retval;
  }

  void reduceWith(PerPackage& data) {
    if (data.empty())
      return;

    PerPackage& l = *m_data.getLocal();
    // TODO parallelize
    for (size_t x = 0; x < data.size(); ++x)
      l[x].second = m_func(l[x].second, data[x].second);
  }

  void localReduce() {
    for (unsigned int i = 1; i < m_data.size(); ++i) {
      if (!runtime::LL::isPackageLeader(i))
        continue;
      reduceWith(*m_data.getRemote(i));
    }
  }

  //-------- Message landing pads ----------

  static void broadcastData(runtime::RecvBuffer& buf) {
    //std::cout << "B: " << galois::runtime::NetworkInterface::ID << "\n";
    DGReducibleVector* dst;
    std::vector<DGReducibleVector*> hosts;
    PerPackage data;
    gDeserialize(buf, hosts, data);
    dst = hosts[galois::runtime::NetworkInterface::ID];
    dst->localUpdate(data);
    dst->hosts = hosts;
  }

  static void registerInstance(runtime::RecvBuffer& buf) {
    assert(galois::runtime::NetworkInterface::ID == 0);
    DGReducibleVector* dst;
    uint32_t host;
    DGReducibleVector* ptr;
    gDeserialize(buf, dst, host, ptr);
    dst->hosts[host] = ptr;
  }

  static void reduceData(runtime::RecvBuffer& buf) {
    //std::cout << "R: " << galois::runtime::NetworkInterface::ID << "\n";
    DGReducibleVector* dst;
    std::vector<DGReducibleVector*> hosts;
    bool reset;
    PerPackage data;
    gDeserialize(buf, hosts, reset, data);
    dst = hosts[galois::runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    dst->reduced++;
    dst->reduceWith(data);
    int expect = expected();
    if (expect == dst->reduced && galois::runtime::NetworkInterface::ID != 0) {
      //std::cout << "r: " << galois::runtime::NetworkInterface::ID << "->" << ((galois::runtime::NetworkInterface::ID - 1)/2) << "\n";
      dst->reduced = 0;
      runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, reset, *dst->m_data.getLocal());
      runtime::getSystemNetworkInterface().send((galois::runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
      if (reset)
        dst->localUpdate();
    }
  }

  static void startReduce(runtime::RecvBuffer& buf) {
    //std::cout << "S: " << galois::runtime::NetworkInterface::ID << "\n";
    std::vector<DGReducibleVector*> hosts;
    bool reset;
    gDeserialize(buf, hosts, reset);
    DGReducibleVector* dst = hosts[galois::runtime::NetworkInterface::ID];
    dst->hosts = hosts;
    dst->localReduce();
    if (expected() == 0) {
      //std::cout << "s: " << galois::runtime::NetworkInterface::ID << "->" << ((galois::runtime::NetworkInterface::ID - 1)/2) << "\n";
      runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, reset, *dst->m_data.getLocal());
      runtime::getSystemNetworkInterface().send((galois::runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  static void startReset(runtime::RecvBuffer& buf) {
    std::vector<DGReducibleVector*> hosts;
    gDeserialize(buf, hosts);
    DGReducibleVector* dst = hosts[galois::runtime::NetworkInterface::ID];
    dst->hosts = hosts;
    dst->localUpdate();
  }

  struct Allocate {
    runtime::gptr<DGReducibleVector> self;
    size_t size;
    Allocate() { }
    Allocate(runtime::gptr<DGReducibleVector> p, size_t s): self(p), size(s) { }

    void operator()(unsigned tid, unsigned) {
      if (!runtime::LL::isPackageLeader(tid))
        return;

      PerPackage& p = *self->m_data.getLocal();

      self->lock.lock();
      p.resize(size, Item(self->m_initial));
      self->m_size = size;
      self->lock.unlock();
    }

    typedef int tt_has_serialize;
    void serialize(runtime::SerializeBuffer& s) const { gSerialize(s, self, size); }
    void deserialize(runtime::DeSerializeBuffer& s) { gDeserialize(s, self, size); }
  };


public:
  void doReduce(bool reset = true) {
    runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts, reset);
    runtime::getSystemNetworkInterface().broadcast(&startReduce, sbuf);

    int expect = expected();
    localReduce();

    while (expect != reduced) {
      assert(runtime::LL::getTID() == 0);
      runtime::getSystemNetworkInterface().handleReceives();
    }
    reduced = 0;
  }

  //! Host 0 returns before broadcast is over
  void doBroadcast() {
    localUpdate(*m_data.getLocal());
    runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts, *m_data.getLocal());
    runtime::getSystemNetworkInterface().broadcast(&broadcastData, sbuf, false);
  }

  void doReset() {
    runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts);
    runtime::getSystemNetworkInterface().broadcast(&startReset, sbuf);

    localUpdate();
  }

  void doAllReduce() {
    doReduce(false);
    doBroadcast();
  }

  T& get(size_t idx) {
    return m_data.getLocal()->at(idx).second;
  }

  void update(size_t i, const T& t) {
    PerPackage& p = *m_data.getLocal();
    int val;
    while (true) {
      val = p[i].first;
      if (val == 0 && __sync_bool_compare_and_swap(&p[i].first, 0, 1)) {
        p[i].second = m_func(p[i].second, t);
        p[i].first = 0;
        break;
      }
    }
  }

  void allocate(size_t size) {
    galois::on_each(Allocate(runtime::gptr<DGReducibleVector>(this), size));
  }

  DGReducibleVector(const BinFunc& f = BinFunc(), const T& initial = T()): 
    m_func(f), m_initial(initial), m_size(0), reduced(0)
  {
    hosts.resize(galois::runtime::NetworkInterface::Num);
    hosts[galois::runtime::NetworkInterface::ID] = this;
    //runtime::allocatePerHost(this);
  }

  DGReducibleVector(galois::runtime::DeSerializeBuffer& s): reduced(0) {
    gDeserialize(s, m_func, m_initial, hosts);
    runtime::SendBuffer buf;
    gSerialize(buf, hosts[0], galois::runtime::NetworkInterface::ID, this);
    runtime::getSystemNetworkInterface().send(0, &registerInstance, buf);
  }

  typedef int tt_is_persistent;
  typedef int tt_has_serialize;

  void serialize(galois::runtime::SerializeBuffer& s) const {
    gSerialize(s, m_func, m_initial, hosts);
  }
};


} //namespace galois

#endif
