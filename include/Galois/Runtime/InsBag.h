/** Insert Bag -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_INSBAG_H
#define GALOIS_RUNTIME_INSBAG_H

#include "Galois/Accumulator.h"
#include "Galois/Runtime/mm/mem.h" 
#include <boost/functional.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <list>


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

  struct splicer {
    void operator()(ListTy& lhs, ListTy& rhs) {
      if (!rhs.empty())
	lhs.splice(lhs.begin(), rhs);
    }
  };

  Galois::GReducible<ListTy, splicer> heads;
  
  
  //GaloisRuntime::MM::TSBlockAlloc<holder> allocSrc;
  GaloisRuntime::MM::FixedSizeAllocator allocSrc;

public:
  galois_insert_bag()
    :allocSrc(sizeof(holder))
  {}

  ~galois_insert_bag() {
    ListTy& L = heads.get();
    while (!L.empty()) {
      holder* H = &L.front();
      L.pop_front();
      allocSrc.deallocate(H);
    }
  }

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
