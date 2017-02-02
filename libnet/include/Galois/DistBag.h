/** Distributed Accumulator type -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */

#ifndef GALOIS_DISTBAG_H
#define GALOIS_DISTBAG_H

namespace Galois {

template<typename ValueTy, typename FunctionTy>
class DGBag {
  Galois::Runtime::NetworkInterface& net = Galois::Runtime::getSystemNetworkInterface();

  const FunctionTy &helper_fn;
  std::string loopName;
  bool didWork;
  std::vector<std::vector<ValueTy>> bagItems_vec;

  static uint32_t num_Hosts_recvd;
  static std::vector<ValueTy> workItem_recv_vec;
  static std::vector<bool> hosts_didWork_vec;

  static void recv_BagItems(uint32_t src, Galois::Runtime::RecvBuffer& buf){
    bool x_didWork;
    unsigned x_ID;
    //XXX: Why pair?
    //std::vector<std::pair<int, int>> vec;
    //TODO: get graphNode type or value type up here.
    std::vector<ValueTy> vec;

    gDeserialize(buf, x_ID, x_didWork, vec);
    workItem_recv_vec.insert(workItem_recv_vec.end(), vec.begin(), vec.end());
    hosts_didWork_vec.push_back(x_didWork);
    //num_Hosts_recvd++;
  }

  void init_sync() {
    num_Hosts_recvd = 0;
    hosts_didWork_vec.clear();
    workItem_recv_vec.clear();
    Galois::Runtime::getHostBarrier().wait();
  }

public: 
  DGBag(const FunctionTy &fn, std::string name) : helper_fn(fn), loopName(name) {
    bagItems_vec.resize(net.Num);
  }

  void set(InsertBag<ValueTy> &bag) {
    //std::string init_str("DISTRIBUTED_BAG_INIT_" + loopName + "_" + std::to_string(helper_fn.get_run_num()));
    std::string init_str("DISTRIBUTED_BAG_INIT_" + loopName + "_" + (helper_fn.get_run_identifier()));
    Galois::StatTimer StatTimer_init(init_str.c_str());
    StatTimer_init.start();
    init_sync();
    didWork = !bag.empty();
    for(auto ii = bag.begin(); ii != bag.end(); ++ii)
    {
      //helper_fn get the hostID for this node.
      bagItems_vec[helper_fn((*ii))].push_back((*ii));
    }
    StatTimer_init.stop();
  }

  void set_local(int* array, size_t size) {
    std::string init_str("DISTRIBUTED_BAG_INIT_" + loopName + "_" + (helper_fn.get_run_identifier()));
    Galois::StatTimer StatTimer_init(init_str.c_str());
    StatTimer_init.start();
    init_sync();
    didWork = (size > 0);
    for (auto i = 0; i < size; ++i) {
      ValueTy ii = helper_fn.getGNode(array[i]);
      bagItems_vec[helper_fn(ii)].push_back(ii);
    }
    StatTimer_init.stop();
  }

  void sync() {
    //std::string sync_str("DISTRIBUTED_BAG_SYNC_" + loopName + "_" + std::to_string(helper_fn.get_run_num()));
    std::string sync_str("DISTRIBUTED_BAG_SYNC_" + loopName + "_" + (helper_fn.get_run_identifier()));
    Galois::StatTimer StatTimer_sync(sync_str.c_str());
    StatTimer_sync.start();

    //std::string work_bytes_str("WORKLIST_BYTES_SENT_" + loopName + "_" + std::to_string(helper_fn.get_run_num()));
    std::string work_bytes_str("WORKLIST_BYTES_SENT_" + loopName + "_" + (helper_fn.get_run_identifier()));
    Galois::Statistic num_work_bytes(work_bytes_str.c_str());
    //send things to other hosts.
    for(auto x = 0U; x < net.Num; ++x){
      if(x == net.ID)
        continue;
      Galois::Runtime::SendBuffer b;
      gSerialize(b, net.ID,didWork, bagItems_vec[x]);
      num_work_bytes += b.size();
      net.sendTagged(x, Galois::Runtime::evilPhase, b);
      //net.sendMsg(x, recv_BagItems, b);
    }
    net.flush();

    //receive
    for(auto x = 0U; x < net.Num; ++x) {
      if(x == net.ID)
        continue;
      decltype(net.recieveTagged(Galois::Runtime::evilPhase,nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      } while (!p);
      recv_BagItems(p->first, p->second);
    }
    ++Galois::Runtime::evilPhase;

    //while(num_Hosts_recvd < (net.Num - 1)){
      //net.handleReceives();
    //}

    workItem_recv_vec.insert(workItem_recv_vec.end(), bagItems_vec[net.ID].begin(), bagItems_vec[net.ID].end());
    std::transform(workItem_recv_vec.begin(), workItem_recv_vec.end(), workItem_recv_vec.begin(), [&](ValueTy i)->ValueTy {return helper_fn.getLocalID(i);});
    std::unique(workItem_recv_vec.begin(), workItem_recv_vec.end());
    //std::string work_item_str("NUM_WORK_ITEMS_" + loopName + "_" + std::to_string(helper_fn.get_run_num()));
    std::string work_item_str("NUM_WORK_ITEMS_" + loopName + "_" + helper_fn.get_run_identifier());
    Galois::Statistic num_work_items(work_item_str.c_str());
    num_work_items += workItem_recv_vec.size();

    assert((hosts_didWork_vec.size() == (net.Num - 1)));
    for(auto x = 0U; x < net.Num; ++x){
      bagItems_vec[x].clear();
    }

    StatTimer_sync.stop();
  }

  static const std::vector<ValueTy> &get() {
    return workItem_recv_vec;
  }

  bool canTerminate() {
    bool terminate = !didWork;
    if(terminate)
      for(auto x : hosts_didWork_vec)
        terminate = (terminate && !x);
    return terminate;
  }
};

template<typename ValueTy, typename FunctionTy>
uint32_t DGBag<ValueTy, FunctionTy>::num_Hosts_recvd = 0;

template<typename ValueTy, typename FunctionTy>
std::vector<ValueTy> DGBag<ValueTy, FunctionTy>::workItem_recv_vec;

template<typename ValueTy, typename FunctionTy>
std::vector<bool> DGBag<ValueTy, FunctionTy>::hosts_didWork_vec;

}
#endif
