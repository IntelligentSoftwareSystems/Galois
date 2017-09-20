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
//FIXME: This is complete bogusness


#include "Galois/Runtime/PerHostStorage.h"

namespace galois {

template<typename T, typename BinFunc>
class DGReducible : public Runtime::Lockable {
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
    if (galois::Runtime::NetworkInterface::ID * 2 + 1 < galois::Runtime::NetworkInterface::Num)
      ++retval;
    if (galois::Runtime::NetworkInterface::ID * 2 + 2 < galois::Runtime::NetworkInterface::Num)
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
  static void broadcastData(Runtime::RecvBuffer& buf) {
    //std::cout << "B: " << galois::Runtime::NetworkInterface::ID << "\n";
    DGReducible* dst;
    std::vector<DGReducible*> hosts;
    T data;
    gDeserialize(buf, hosts, data);
    dst = hosts[galois::Runtime::NetworkInterface::ID];
    assert(dst);
    dst->localReset(data);
    dst->hosts = hosts;
  }

  static void registerInstance(Runtime::RecvBuffer& buf) {
    assert(galois::Runtime::NetworkInterface::ID == 0);
    DGReducible* dst;
    uint32_t host;
    DGReducible* ptr;
    gDeserialize(buf, dst, host, ptr);
    dst->hosts[host] = ptr;
  }

  static void reduceData(Runtime::RecvBuffer& buf) {
    //std::cout << "R: " << galois::Runtime::NetworkInterface::ID << "\n";
    DGReducible* dst;
    std::vector<DGReducible*> hosts;
    T data;
    gDeserialize(buf, hosts, data);
    dst = hosts[galois::Runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    dst->reduced++;
    dst->reduceWith(data);
    int expect = expected();
    if (expect == dst->reduced && galois::Runtime::NetworkInterface::ID != 0) {
      //std::cout << "r: " << galois::Runtime::NetworkInterface::ID << "->" << ((galois::Runtime::NetworkInterface::ID - 1)/2) << "\n";
      dst->reduced = 0;
      Runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->r_data);
      dst->r_data = dst->m_initial; //Reset reduce buffer
      Runtime::getSystemNetworkInterface().send((galois::Runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  static void startReduce(Runtime::RecvBuffer& buf) {
    //std::cout << "S: " << galois::Runtime::NetworkInterface::ID << "\n";
    std::vector<DGReducible*> hosts;
    gDeserialize(buf, hosts);
    DGReducible* dst = hosts[galois::Runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    dst->localReduce();
    if (expected() == 0) {
      //std::cout << "s: " << galois::Runtime::NetworkInterface::ID << "->" << ((galois::Runtime::NetworkInterface::ID - 1)/2) << "\n";
      Runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->r_data);
      Runtime::getSystemNetworkInterface().send((galois::Runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  T& doReduce() {
    //Send reduce messages
    Runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts);
    Runtime::getSystemNetworkInterface().broadcast(&startReduce, sbuf);

    int expect = expected();
    r_data = m_initial;
    localReduce();

    while (expect != reduced) {
      //spin processing network packets
      assert(Runtime::LL::getTID() == 0);
      Runtime::getSystemNetworkInterface().handleReceives();
    }
    reduced = 0;
    return r_data;
  }

  void doBroadcast(const T& data) {
    localReset(data);
    Runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts, data);
    Runtime::getSystemNetworkInterface().broadcast(&broadcastData, sbuf, false);
  }

  T& get() {
    return *m_data.getLocal();
  }

  DGReducible(const BinFunc& f = BinFunc(), const T& initial = T()) :m_func(f), m_initial(initial), reduced(0) {
    hosts.resize(galois::Runtime::NetworkInterface::Num);
    hosts[galois::Runtime::NetworkInterface::ID] = this;
    localReset(m_initial);
  }

  DGReducible(galois::Runtime::DeSerializeBuffer& s) :reduced(0) {
    gDeserialize(s, m_func, m_initial, hosts);
    localReset(m_initial);
    Runtime::SendBuffer buf;
    gSerialize(buf, hosts[0], galois::Runtime::NetworkInterface::ID, this);
    Runtime::getSystemNetworkInterface().send(0, &registerInstance, buf);
  }

  // mark the graph as persistent so that it is distributed
  typedef int tt_is_persistent;
  typedef int tt_has_serialize;
  void serialize(galois::Runtime::SerializeBuffer& s) const {
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
    if (galois::Runtime::NetworkInterface::ID * 2 + 1 < galois::Runtime::NetworkInterface::Num)
      ++retval;
    if (galois::Runtime::NetworkInterface::ID * 2 + 2 < galois::Runtime::NetworkInterface::Num)
      ++retval;
    return retval;
  }

public:
  static void broadcastData(Runtime::RecvBuffer& buf) {
    //std::cout << "B: " << galois::Runtime::NetworkInterface::ID << "\n";
    DGReducibleInplace* dst;
    std::vector<DGReducibleInplace*> hosts;
    T data;
    gDeserialize(buf, hosts);
    dst = hosts[galois::Runtime::NetworkInterface::ID];
    gDeserialize(buf, dst->m_data);
    dst->doBroadcast(data);
  }

  static void registerInstance(Runtime::RecvBuffer& buf) {
    assert(galois::Runtime::NetworkInterface::ID == 0);
    DGReducibleInplace* dst;
    uint32_t host;
    DGReducibleInplace* ptr;
    gDeserialize(buf, dst, host, ptr);
    dst->hosts[host] = ptr;
  }

  static void reduceData(Runtime::RecvBuffer& buf) {
    //std::cout << "R: " << galois::Runtime::NetworkInterface::ID << "\n";
    DGReducibleInplace* dst;
    std::vector<DGReducibleInplace*> hosts;
    T data;
    gDeserialize(buf, hosts, data);
    dst = hosts[galois::Runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    dst->reduced++;
    dst->m_func(dst->m_data, data);
    int expect = expected();
    if (expect == dst->reduced && galois::Runtime::NetworkInterface::ID != 0) {
      //std::cout << "r: " << galois::Runtime::NetworkInterface::ID << "->" << ((galois::Runtime::NetworkInterface::ID - 1)/2) << "\n";
      dst->reduced = 0;
      Runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->m_data);
      Runtime::getSystemNetworkInterface().send((galois::Runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  static void startReduce(Runtime::RecvBuffer& buf) {
    //std::cout << "S: " << galois::Runtime::NetworkInterface::ID << "\n";
    std::vector<DGReducibleInplace*> hosts;
    gDeserialize(buf, hosts);
    DGReducibleInplace* dst = hosts[galois::Runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    if (expected() == 0) {
      //std::cout << "s: " << galois::Runtime::NetworkInterface::ID << "->" << ((galois::Runtime::NetworkInterface::ID - 1)/2) << "\n";
      Runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, dst->m_data);
      Runtime::getSystemNetworkInterface().send((galois::Runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  T& doReduce() {
    //Send reduce messages
    Runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts);
    Runtime::getSystemNetworkInterface().broadcast(&startReduce, sbuf);

    int expect = expected();

    while (expect != reduced) {
      //spin processing network packets
      assert(Runtime::LL::getTID() == 0);
      Runtime::getSystemNetworkInterface().handleReceives();
    }
    reduced = 0;
    return m_data;
  }

  T& get() {
    return m_data;
  }

  DGReducibleInplace(const BinFunc& f = BinFunc()) :m_func(f), reduced(0) {
    hosts.resize(galois::Runtime::NetworkInterface::Num);
    hosts[galois::Runtime::NetworkInterface::ID] = this;
  }

  DGReducibleInplace(galois::Runtime::DeSerializeBuffer& s) :reduced(0) {
    gDeserialize(s, m_func, hosts);
    Runtime::SendBuffer buf;
    gSerialize(buf, hosts[0], galois::Runtime::NetworkInterface::ID, this);
    Runtime::getSystemNetworkInterface().send(0, &registerInstance, buf);
  }

  // mark the graph as persistent so that it is distributed
  typedef int tt_is_persistent;
  typedef int tt_has_serialize;
  void serialize(galois::Runtime::SerializeBuffer& s) const {
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

    Item(galois::Runtime::DeSerializeBuffer& s) {
      galois::Runtime::gDeserialize(s, second);
      first = 0;
    }

    typedef int tt_has_serialize;
    void serialize(galois::Runtime::SerializeBuffer& s) const {
      galois::Runtime::gSerialize(s, second);
    }

    void deserialize(galois::Runtime::DeSerializeBuffer& s) {
      galois::Runtime::gDeserialize(s, second);
      first = 0;
    }
  };

  typedef std::vector<Item> PerPackage;
  typedef Runtime::PerPackageStorage<PerPackage> Data;

  BinFunc m_func;
  T m_initial;
  Data m_data;
  Runtime::LL::SimpleLock lock;

  std::vector<DGReducibleVector*> hosts;
  size_t m_size;

  volatile int reduced;

  void localUpdate(const PerPackage& init) {
    for (unsigned int i = 0; i < m_data.size(); ++i) {
      if (!Runtime::LL::isPackageLeader(i))
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
      if (!Runtime::LL::isPackageLeader(i))
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
    if (galois::Runtime::NetworkInterface::ID * 2 + 1 < galois::Runtime::NetworkInterface::Num)
      ++retval;
    if (galois::Runtime::NetworkInterface::ID * 2 + 2 < galois::Runtime::NetworkInterface::Num)
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
      if (!Runtime::LL::isPackageLeader(i))
        continue;
      reduceWith(*m_data.getRemote(i));
    }
  }

  //-------- Message landing pads ----------

  static void broadcastData(Runtime::RecvBuffer& buf) {
    //std::cout << "B: " << galois::Runtime::NetworkInterface::ID << "\n";
    DGReducibleVector* dst;
    std::vector<DGReducibleVector*> hosts;
    PerPackage data;
    gDeserialize(buf, hosts, data);
    dst = hosts[galois::Runtime::NetworkInterface::ID];
    dst->localUpdate(data);
    dst->hosts = hosts;
  }

  static void registerInstance(Runtime::RecvBuffer& buf) {
    assert(galois::Runtime::NetworkInterface::ID == 0);
    DGReducibleVector* dst;
    uint32_t host;
    DGReducibleVector* ptr;
    gDeserialize(buf, dst, host, ptr);
    dst->hosts[host] = ptr;
  }

  static void reduceData(Runtime::RecvBuffer& buf) {
    //std::cout << "R: " << galois::Runtime::NetworkInterface::ID << "\n";
    DGReducibleVector* dst;
    std::vector<DGReducibleVector*> hosts;
    bool reset;
    PerPackage data;
    gDeserialize(buf, hosts, reset, data);
    dst = hosts[galois::Runtime::NetworkInterface::ID];
    assert(dst);
    dst->hosts = hosts;
    dst->reduced++;
    dst->reduceWith(data);
    int expect = expected();
    if (expect == dst->reduced && galois::Runtime::NetworkInterface::ID != 0) {
      //std::cout << "r: " << galois::Runtime::NetworkInterface::ID << "->" << ((galois::Runtime::NetworkInterface::ID - 1)/2) << "\n";
      dst->reduced = 0;
      Runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, reset, *dst->m_data.getLocal());
      Runtime::getSystemNetworkInterface().send((galois::Runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
      if (reset)
        dst->localUpdate();
    }
  }

  static void startReduce(Runtime::RecvBuffer& buf) {
    //std::cout << "S: " << galois::Runtime::NetworkInterface::ID << "\n";
    std::vector<DGReducibleVector*> hosts;
    bool reset;
    gDeserialize(buf, hosts, reset);
    DGReducibleVector* dst = hosts[galois::Runtime::NetworkInterface::ID];
    dst->hosts = hosts;
    dst->localReduce();
    if (expected() == 0) {
      //std::cout << "s: " << galois::Runtime::NetworkInterface::ID << "->" << ((galois::Runtime::NetworkInterface::ID - 1)/2) << "\n";
      Runtime::SendBuffer sbuf;
      gSerialize(sbuf, hosts, reset, *dst->m_data.getLocal());
      Runtime::getSystemNetworkInterface().send((galois::Runtime::NetworkInterface::ID - 1)/2, &reduceData, sbuf);
    }
  }

  static void startReset(Runtime::RecvBuffer& buf) {
    std::vector<DGReducibleVector*> hosts;
    gDeserialize(buf, hosts);
    DGReducibleVector* dst = hosts[galois::Runtime::NetworkInterface::ID];
    dst->hosts = hosts;
    dst->localUpdate();
  }

  struct Allocate {
    Runtime::gptr<DGReducibleVector> self;
    size_t size;
    Allocate() { }
    Allocate(Runtime::gptr<DGReducibleVector> p, size_t s): self(p), size(s) { }

    void operator()(unsigned tid, unsigned) {
      if (!Runtime::LL::isPackageLeader(tid))
        return;

      PerPackage& p = *self->m_data.getLocal();

      self->lock.lock();
      p.resize(size, Item(self->m_initial));
      self->m_size = size;
      self->lock.unlock();
    }

    typedef int tt_has_serialize;
    void serialize(Runtime::SerializeBuffer& s) const { gSerialize(s, self, size); }
    void deserialize(Runtime::DeSerializeBuffer& s) { gDeserialize(s, self, size); }
  };


public:
  void doReduce(bool reset = true) {
    Runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts, reset);
    Runtime::getSystemNetworkInterface().broadcast(&startReduce, sbuf);

    int expect = expected();
    localReduce();

    while (expect != reduced) {
      assert(Runtime::LL::getTID() == 0);
      Runtime::getSystemNetworkInterface().handleReceives();
    }
    reduced = 0;
  }

  //! Host 0 returns before broadcast is over
  void doBroadcast() {
    localUpdate(*m_data.getLocal());
    Runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts, *m_data.getLocal());
    Runtime::getSystemNetworkInterface().broadcast(&broadcastData, sbuf, false);
  }

  void doReset() {
    Runtime::SendBuffer sbuf;
    gSerialize(sbuf, hosts);
    Runtime::getSystemNetworkInterface().broadcast(&startReset, sbuf);

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
    galois::on_each(Allocate(Runtime::gptr<DGReducibleVector>(this), size));
  }

  DGReducibleVector(const BinFunc& f = BinFunc(), const T& initial = T()): 
    m_func(f), m_initial(initial), m_size(0), reduced(0)
  {
    hosts.resize(galois::Runtime::NetworkInterface::Num);
    hosts[galois::Runtime::NetworkInterface::ID] = this;
    //Runtime::allocatePerHost(this);
  }

  DGReducibleVector(galois::Runtime::DeSerializeBuffer& s): reduced(0) {
    gDeserialize(s, m_func, m_initial, hosts);
    Runtime::SendBuffer buf;
    gSerialize(buf, hosts[0], galois::Runtime::NetworkInterface::ID, this);
    Runtime::getSystemNetworkInterface().send(0, &registerInstance, buf);
  }

  typedef int tt_is_persistent;
  typedef int tt_has_serialize;

  void serialize(galois::Runtime::SerializeBuffer& s) const {
    gSerialize(s, m_func, m_initial, hosts);
  }
};


} //namespace galois

#endif
