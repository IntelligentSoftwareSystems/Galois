// TODO : move to experimental if it won't compile / is uncessary
#include "galois/Bag.h"
#include "galois/runtime/Serialize.h"
#include <boost/range/iterator_range.hpp>
#include <iostream>
#include <algorithm>
#include <sstream>

// TODO: move to libdist or delete
namespace galois {
namespace worklists {

template <typename WL, typename GraphTy>
class WLdistributed : public WL {
  // typedef typename WL::value_type workItemTy;
  typedef typename WL::value_type value_type;
  std::vector<galois::InsertBag<uint64_t>> bag_vec;
  // std::vector<std::vector<uint64_t>> bag_vec;
  WL workList;
  std::deque<uint64_t> local_wl;
  GraphTy& graph;
  bool canTerminate;

  // uint64_t c;

public:
  WLdistributed(GraphTy& _graph) : workList(), graph(_graph) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    bag_vec.resize(net.Num);
  }

  void push(const typename WL::value_type& v) {
    workList.push(v);
    // c++;
  }

  template <typename I>
  void push(I b, I e) {
    workList.push(b, e);
    // c++;
  }

  /**
   * Convert ids in local worklist to local ids from global ids.
   */
  void convertGID2LID_localWL() {
    uint32_t num = size();
    galois::do_all(galois::iterate(0u, num),
                   [&](uint32_t n) { local_wl[n] = graph.G2L(local_wl[n]); },
                   galois::loopname("G2L_wl"));
  }

  void insertToLocalWL(galois::InsertBag<uint64_t>& bag) {
    local_wl.insert(local_wl.end(), bag.begin(), bag.end());
    // assert(local_wl.size() == totalSize);
    bag.clear();
  }

  void insertToLocalWL(std::deque<uint64_t>& d) {
    local_wl.insert(local_wl.end(), d.begin(), d.end());
  }

  void sync() {
    auto& net = galois::runtime::getSystemNetworkInterface();
    std::string statSendBytes_str("WL_SEND_BYTES_" +
                                  graph.get_run_identifier());
    std::string timer_wl_send_recv_str("WL_SYNC_TIME_SEND_RECV_" +
                                       graph.get_run_identifier());
    std::string timer_wl_total_str("WL_SYNC_TIME_TOTAL_" +
                                   graph.get_run_identifier());

    galois::StatTimer StatTimer_total_sync(timer_wl_total_str.c_str());
    galois::StatTimer StatTimer_send_recv(timer_wl_send_recv_str.c_str());
    galois::Statistic send_bytes(statSendBytes_str);

    StatTimer_total_sync.start();

    // Did work in the previous round
    if (local_wl.size() > 0)
      canTerminate = false;
    else
      canTerminate = true;

    clear_wl();

    // sort out location of hosts for things that were pushed into our
    // worklist
    galois::on_each(
        [&](const unsigned tid, const unsigned numT) {
          galois::optional<value_type> p;
          while (p = workList.pop()) {
            auto GID = graph.getGID(*p);
            assert(graph.getHostID(GID) < bag_vec.size());
            bag_vec[graph.getHostID(GID)].push(GID);
          }
        },
        galois::loopname("worklist bags"));
    //    uint64_t count = 0;
    //    galois::optional<value_type> p;
    //    while (p = workList.pop()) {
    //      count++;
    //      auto GID = graph.getGID(*p);
    //      assert(graph.getHostID(GID) < bag_vec.size());
    //      bag_vec[graph.getHostID(GID)].push_back(GID);
    //    }

    //    uint64_t new_count = 0;

    //    for (auto i = bag_vec[0].begin(); i != bag_vec[0].end(); i++) {
    //      new_count++;
    //    }

    //    printf("pushed is %lu\n", c);
    //    printf("in bag = %lu\n", count);
    //    printf("bag size = %lu\n", bag_vec[0].size());
    //    printf("bag size 2= %lu\n", new_count);

    //    c = 0;

    StatTimer_send_recv.start();

    for (auto h = 0; h < net.Num; ++h) {
      // if any bag has things, we cannot terminate
      if (canTerminate && (bag_vec[h].size() > 0)) {
        canTerminate = false;
      }

      // insert our own items to our own list
      if (h == net.ID) {
        insertToLocalWL(bag_vec[h]);
        continue;
      }

      // otherwise send off itmes to appropriate host
      size_t bagSize = bag_vec[h].size();
      galois::runtime::SendBuffer b;
      galois::runtime::gSerialize(b, canTerminate);
      galois::runtime::gSerialize(b, bag_vec[h]);
      send_bytes += b.size();
      // assert(b.size() == sizeof(uint64_t)*(bagSize + 2));
      // std::stringstream ss;
      // ss << net.ID << " ] : SENDING : " << bagSize << "\n";
      // std::cerr << ss.str();
      net.sendTagged(h, galois::runtime::evilPhase, b);
    }
    net.flush();

    // receive
    for (auto h = 0; h < net.Num; ++h) {
      if ((h == net.ID))
        continue;

      decltype(net.recieveTagged(galois::runtime::evilPhase, nullptr)) p;

      do {
        p = net.recieveTagged(galois::runtime::evilPhase, nullptr);
      } while (!p);

      bool other_canTerminate = true;
      galois::runtime::gDeserialize(p->second, other_canTerminate);
      canTerminate = (other_canTerminate && canTerminate);

      // get stuff, insert to our list
      std::deque<uint64_t> s;
      // TODO avoid double copy
      galois::runtime::gDeserialize(p->second, s);
      insertToLocalWL(s);
      // std::stringstream ss;
      // ss << net.ID << " ] : RECV : " << s.size() << " other CT : " <<
      // other_canTerminate << "\n"; std::cerr << ss.str();
    }
    ++galois::runtime::evilPhase;
    StatTimer_send_recv.stop();

    // Sort and remove duplicates from the worklist.
    std::sort(local_wl.begin(), local_wl.end());
    std::unique(local_wl.begin(), local_wl.end());

    // TODO Check that duplicates are being removed
    // assert(local_wl.size() <= graph.numOwned)

    // Convert Global to Local IDs
    convertGID2LID_localWL();
    StatTimer_total_sync.stop();
  }

  bool can_terminate() {
    // return false;
    return canTerminate;
  }

  template <typename Ty>
  void push_initial(Ty e) {
    local_wl.push_back(e);
  }

  template <typename Itr>
  void push_initial(Itr b, Itr e) {
    local_wl.assign(b, e);
  }

  galois::optional<value_type> pop() { return local_wl.pop_back(); }

  auto getRange() {
    return boost::make_iterator_range(local_wl.begin(), local_wl.end());
  }

  auto clear_wl() { local_wl.clear(); }

  bool empty() const { return local_wl.empty(); }

  size_t size() const { return local_wl.size(); }
  size_t size2() const { return workList.size(); }
};

} // end namespace worklists
} // end namespace galois
