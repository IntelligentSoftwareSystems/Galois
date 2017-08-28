#include "Galois/Bag.h"
#include "Galois/Runtime/Serialize.h"
#include <boost/range/iterator_range.hpp>
#include<iostream>
#include <algorithm>

//TODO: move to libdist or delete
namespace Galois {
namespace WorkList {


template <typename WL, typename GraphTy>
class WLdistributed: public WL {

  //typedef typename WL::value_type workItemTy;
  typedef typename WL::value_type value_type;
  std::vector<Galois::InsertBag<uint64_t>> bag_vec;
  //std::vector<std::vector<uint64_t>> bag_vec;
  WL workList;
  std::deque<uint64_t> local_wl;
  GraphTy& graph;

public:

  WLdistributed (GraphTy& _graph): workList (),graph(_graph) {
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    bag_vec.resize(net.Num);
  }

  void push (const typename WL::value_type& v) {
    workList.push(v);
  }

  template <typename I>
  void push (I b, I e) {
    workList.push(b,e);
  }

  void insertToLocalWL(Galois::InsertBag<uint64_t>& bag){
    auto headerVec = bag.getHeads();
    size_t totalSize = bag.size();
    for(auto head : headerVec){
      local_wl.insert(local_wl.end(), head->dbegin, head->dend);
    }
    assert(local_wl.size() == totalSize);
    bag.clear();
  }

  void insertToLocalWL(std::deque<uint64_t>& d){
    local_wl.insert(local_wl.end(), d.begin(), d.end());
  }

  void sync(){
    clear_wl();
    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::on_each([&] (const unsigned tid, const unsigned numT){
        Galois::optional<value_type> p;
        while(p = workList.pop()){
        auto GID = graph.getGID(*p);
        assert(graph.getHostID(GID) < bag_vec.size());
        if(graph.id == 0 && graph.getHostID(GID) == 1){
          std::cout << "FOUND\n";
        }
        bag_vec[graph.getHostID(GID)].push(GID);
        }
        }, Galois::loopname("worklist bags"));

    for(auto h = 0; h < net.Num; ++h){
      if(h == net.ID){
        insertToLocalWL(bag_vec[h]);
        continue;
      }

      size_t bagSize =  bag_vec[h].size();
      Galois::Runtime::SendBuffer b;
      Galois::Runtime::gSerialize(b, bag_vec[h]);
      assert(b.size() == sizeof(uint64_t)*(bagSize + 1));
      std::cerr << net.ID << " ] : SENDING : " << bagSize << "\n";
      net.sendTagged(h, Galois::Runtime::evilPhase, b);
    }
    net.flush();

    //receive
    for (auto h = 0; h < net.Num; ++h) {
      if ((h == net.ID))
        continue;
      decltype(net.recieveTagged(Galois::Runtime::evilPhase,nullptr)) p;
      do {
        net.handleReceives();
        p = net.recieveTagged(Galois::Runtime::evilPhase, nullptr);
      } while (!p);
      std::deque<uint64_t> s;
      //TODO avoid double copy
      Galois::Runtime::gDeserialize(p->second,s);
      insertToLocalWL(s);
      std::cerr << net.ID << " ] : RECV : " << s.size() << "\n";
    }
    ++Galois::Runtime::evilPhase;

    //Sort and remove duplicates from the worklist.
    std::sort(local_wl.begin(), local_wl.end());
    std::unique(local_wl.begin(), local_wl.end());

  }

  //template<typename R>
    //void push_initial(const R& range) {
      //local_wl.assign(range.begin(), range.end());
    //}

  template<typename Ty>
    void push_initial(Ty e) {
      local_wl.push_back(e);
    }

  Galois::optional<value_type> pop() {
    return local_wl.pop_back();
  }

  auto getRange() {
    return boost::make_iterator_range(local_wl.begin(), local_wl.end());
  }

  auto clear_wl() {
    local_wl.clear();
  }

  bool empty() const {
    return local_wl.empty();
  }

};


} // end namespace WorkList
} // end namespace Galois

