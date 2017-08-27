#include "Galois/Bag.h"
#include "Galois/Runtime/Serialize.h"
#include<iostream>

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

  //! Not working: Something wrong
  //template <typename _T>
  //struct retype {
    //typedef WLdistributed<typename WL::template retype<_T>> type;
    //typedef WLdistributed<typename WL::template retype<_T>::type> type;
  //};

  //typedef typename WL::value_type value_type;

  //WLdistributed (Galois::InsertBag<value_type1>* b): workList (), bag(b) {}
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
    //for(I i = b; i != e; ++i){
      //bag->push(*i);
    //}

/*    auto& net = Galois::Runtime::getSystemNetworkInterface();
    for(I i = b; i != e; ++i){
      auto hostID = graph.getHostID(graph.getGID((*i).first));
      if(net.ID == hostID){
        workList.push(i,i);
        std::cout << net.ID << "<-- hostID PUSH b,e -> "<< (*i).first << "\n";
      }
      else{
        std::cout << net.ID << "<--  in bag push --> " << (*i).first << "\n";
        bag.push(*i);
      }
    }
*/
  }

  void sync(){

    auto& net = Galois::Runtime::getSystemNetworkInterface();
    Galois::on_each([&] (const unsigned tid, const unsigned numT){
      Galois::optional<value_type> p;
      while(p = workList.pop()){
        auto GID = graph.getGID(*p);
        assert(graph.getHostID(GID) < bag_vec.size());
        bag_vec[graph.getHostID(GID)].push(GID);
      }
    }, Galois::loopname("worklist bags"));


    for(auto h = 0; h < net.Num; ++h){
      if(h == net.ID)
        continue;

      Galois::Runtime::SendBuffer b;
      for(auto i = bag_vec[h].begin(), i_end = bag_vec[h].end(); i != i_end; ++i){
        b.push(*i);
      }

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
      std::deque<uint64_t> d;
      Galois::Runtime::gDeserialize(p->second,d);
      std::cerr << net.ID << " ] " << d.size() << "\n";

    }
    ++Galois::Runtime::evilPhase;
  }

  template<typename R>
    void push_initial(const R& range) {
      workList.push_initial(range);
    }

  Galois::optional<value_type> pop() {
    //return workList.pop();
  }

};


} // end namespace WorkList
} // end namespace Galois

