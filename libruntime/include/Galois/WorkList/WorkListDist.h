#include "Galois/Bag.h"
#include<iostream>

//TODO: move to libdist or delete
namespace Galois {
namespace WorkList {


template <typename WL>
class WLdistributed: public WL {

  //typedef typename WL::value_type workItemTy;
  typedef typename WL::value_type value_type1;
  Galois::InsertBag<value_type1>* bag;
  WL workList;

public:

  //! Not working: Something wrong
  template <typename _T>
  struct retype {
    typedef WLdistributed<typename WL::template retype<_T>> type;
    //typedef WLdistributed<typename WL::template retype<_T>::type> type;
  };

  typedef typename WL::value_type value_type;

  WLdistributed (Galois::InsertBag<value_type>* b): workList (), bag(b) {}

  void push (const typename WL::value_type& v) {
    bag->push(v);
  }

  template <typename I>
  void push (I b, I e) {
    for(I i = b; i != e; ++i){
      bag->push(*i);
    }

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

  template<typename R>
  void push_initial(const R& range) {
    workList.push_initial(range);
  }

  Galois::optional<value_type> pop() {
    return workList.pop();
  }

};


} // end namespace WorkList
} // end namespace Galois

