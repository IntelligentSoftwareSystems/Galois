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

#ifndef GALOIS_DISTBAG_H
#define GALOIS_DISTBAG_H

namespace galois {

template <typename ValueTy, typename FunctionTy>
class DGBag {
  galois::runtime::NetworkInterface& net =
      galois::runtime::getSystemNetworkInterface();

  const FunctionTy& helper_fn;
  std::string loopName;
  bool didWork;
  std::vector<std::vector<ValueTy>> bagItems_vec;

  static uint32_t num_Hosts_recvd;
  static std::vector<ValueTy> workItem_recv_vec;
  static std::vector<bool> hosts_didWork_vec;

  static void recv_BagItems(uint32_t src, galois::runtime::RecvBuffer& buf) {
    bool x_didWork;
    unsigned x_ID;
    // XXX: Why pair?
    // std::vector<std::pair<int, int>> vec;
    // TODO: get graphNode type or value type up here.
    std::vector<ValueTy> vec;

    gDeserialize(buf, x_ID, x_didWork, vec);
    workItem_recv_vec.insert(workItem_recv_vec.end(), vec.begin(), vec.end());
    hosts_didWork_vec.push_back(x_didWork);
    // num_Hosts_recvd++;
  }

  void init_sync() {
    num_Hosts_recvd = 0;
    hosts_didWork_vec.clear();
    workItem_recv_vec.clear();
    galois::runtime::getHostBarrier().wait();
  }

public:
  DGBag(const FunctionTy& fn, std::string name)
      : helper_fn(fn), loopName(name) {
    bagItems_vec.resize(net.Num);
  }

  void set(InsertBag<ValueTy>& bag) {
    // std::string init_str("DISTRIBUTED_BAG_INIT_" + loopName + "_" +
    // std::to_string(helper_fn.get_run_num()));
    std::string init_str("DISTRIBUTED_BAG_INIT_" + loopName + "_" +
                         (helper_fn.get_run_identifier()));
    galois::StatTimer StatTimer_init(init_str.c_str());
    StatTimer_init.start();
    init_sync();
    didWork = !bag.empty();
    for (auto ii = bag.begin(); ii != bag.end(); ++ii) {
      // helper_fn get the hostID for this node.
      bagItems_vec[helper_fn((*ii))].push_back((*ii));
    }
    StatTimer_init.stop();
  }

  void set_local(int* array, size_t size) {
    std::string init_str("DISTRIBUTED_BAG_INIT_" + loopName + "_" +
                         (helper_fn.get_run_identifier()));
    galois::StatTimer StatTimer_init(init_str.c_str());
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
    // std::string sync_str("DISTRIBUTED_BAG_SYNC_" + loopName + "_" +
    // std::to_string(helper_fn.get_run_num()));
    std::string sync_str("DISTRIBUTED_BAG_SYNC_" + loopName + "_" +
                         (helper_fn.get_run_identifier()));
    galois::StatTimer StatTimer_sync(sync_str.c_str());
    StatTimer_sync.start();

    // std::string work_bytes_str("WORKLIST_BYTES_SENT_" + loopName + "_" +
    // std::to_string(helper_fn.get_run_num()));
    std::string work_bytes_str("WORKLIST_BYTES_SENT_" + loopName + "_" +
                               (helper_fn.get_run_identifier()));
    galois::Statistic num_work_bytes(work_bytes_str.c_str());
    // send things to other hosts.
    for (auto x = 0U; x < net.Num; ++x) {
      if (x == net.ID)
        continue;
      galois::runtime::SendBuffer b;
      gSerialize(b, net.ID, didWork, bagItems_vec[x]);
      num_work_bytes += b.size();
      net.sendTagged(x, galois::runtime::evilPhase, b);
    }
    net.flush();

    // receive
    for (auto x = 0U; x < net.Num; ++x) {
      if (x == net.ID)
        continue;
      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;
      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);
      recv_BagItems(p->first, p->second);
    }
    ++galois::runtime::evilPhase;

    workItem_recv_vec.insert(workItem_recv_vec.end(),
                             bagItems_vec[net.ID].begin(),
                             bagItems_vec[net.ID].end());
    std::transform(workItem_recv_vec.begin(), workItem_recv_vec.end(),
                   workItem_recv_vec.begin(), [&](ValueTy i) -> ValueTy {
                     return helper_fn.getLocalID(i);
                   });
    std::unique(workItem_recv_vec.begin(), workItem_recv_vec.end());
    // std::string work_item_str("NUM_WORK_ITEMS_" + loopName + "_" +
    // std::to_string(helper_fn.get_run_num()));
    std::string work_item_str("NUM_WORK_ITEMS_" + loopName + "_" +
                              helper_fn.get_run_identifier());
    galois::Statistic num_work_items(work_item_str.c_str());
    num_work_items += workItem_recv_vec.size();

    assert((hosts_didWork_vec.size() == (net.Num - 1)));
    for (auto x = 0U; x < net.Num; ++x) {
      bagItems_vec[x].clear();
    }

    StatTimer_sync.stop();
  }

  static const std::vector<ValueTy>& get() { return workItem_recv_vec; }

  bool canTerminate() {
    bool terminate = !didWork;
    if (terminate)
      for (auto x : hosts_didWork_vec)
        terminate = (terminate && !x);
    return terminate;
  }
};

template <typename ValueTy, typename FunctionTy>
uint32_t DGBag<ValueTy, FunctionTy>::num_Hosts_recvd = 0;

template <typename ValueTy, typename FunctionTy>
std::vector<ValueTy> DGBag<ValueTy, FunctionTy>::workItem_recv_vec;

template <typename ValueTy, typename FunctionTy>
std::vector<bool> DGBag<ValueTy, FunctionTy>::hosts_didWork_vec;

} // namespace galois
#endif
