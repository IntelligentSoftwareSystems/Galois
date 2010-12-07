// Insert Bag implementation -*- C++ -*-
// This one is suppose to be scalable and uses Galois PerCPU classes
#ifndef GALOIS_INSERT_BAG_H_
#define GALOIS_INSERT_BAG_H_

#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/mm/mem.h" 
//#include "Galois/Runtime/mm/BlockAlloc.h"
#include <list>

//#include <boost/intrusive/list.hpp>
//#include <boost/iterator/transform_iterator.hpp>

namespace GaloisRuntime {
  
template< class T>
class galois_insert_bag {
  
  class InsBagTag;
  typedef boost::intrusive::list_base_hook<boost::intrusive::tag<InsBagTag>,
   					   boost::intrusive::link_mode<boost::intrusive::normal_link>
					   > InsBagBaseHook;

  struct holder : public InsBagBaseHook {
    T data;
    T& getData() { return data; }
  };

  typedef boost::intrusive::list<holder,
   				  boost::intrusive::base_hook<InsBagBaseHook>,
   				  boost::intrusive::constant_time_size<false>
   				  > ListTy;
   CPUSpaced<ListTy> heads;
  
  
  //GaloisRuntime::MM::TSBlockAlloc<holder> allocSrc;
  GaloisRuntime::MM::ThreadAwarePrivateHeap<GaloisRuntime::MM::BlockAlloc<sizeof(holder), GaloisRuntime::MM::SystemBaseAlloc> > allocSrc;

  static void merge(ListTy& lhs, ListTy& rhs) {
    if (!rhs.empty())
      lhs.splice(lhs.begin(), rhs);
  }

public:
  galois_insert_bag()
    :heads(merge)
  {}

  typedef T        value_type;
  typedef const T& const_reference;
  typedef T&       reference;

  typedef typename boost::transform_iterator<boost::mem_fun_ref_t<T&,holder>, typename ListTy::iterator> iterator;

  iterator begin() {
    return boost::make_transform_iterator(heads.get().begin(), boost::mem_fun_ref(&holder::getData));
  }
  iterator end() {
    return boost::make_transform_iterator(heads.get().end(), boost::mem_fun_ref(&holder::getData));
  }

  //Only this is thread safe
  reference push(const T& val) {
    holder* h = (holder*)allocSrc.allocate(sizeof(holder));
    new ((void *)&h->data) T(val);
    heads.get().push_front(*h);
    return h->data;
  }
  
};
}
#endif
