// Insert Bag implementation -*- C++ -*-
// This one is suppose to be scalable and uses Galois PerCPU classes
#ifndef GALOIS_INSERT_BAG_H_
#define GALOIS_INSERT_BAG_H_

#include "Galois/Runtime/PerCPU.h"

namespace GaloisRuntime {
  
  template< class T>
  class galois_insert_bag {
    struct bag_item {
      T item;
      bag_item* next;
      
      explicit bag_item(const T& V)
	:item(V), next(0)
      {}
    };
    
    PerCPUPtr<bag_item*> heads;

    void i_push(bag_item* B) {
      bag_item* head = heads.get();
      B->next = head;
      heads.set_assert(B, head);
    }

    //Handle iterator support
    friend class iterator;
    bag_item* getHeadFor(int i) {
      return heads.getRemote(i);
    }

  public:
    typedef T        value_type;
    typedef const T& const_reference;
    typedef T&       reference;
    
    ~galois_insert_bag() {
      for (int i = 0; i <= getMaxThreadID(); ++i) {
	bag_item* head = heads.getRemote(i);
	while (head) {
	  bag_item* n = head;
	  head = n->next;
	  delete n;
	}
      }
    }

    class iterator {
      bag_item* b;
      int cThreadID;
      galois_insert_bag* parent;
      friend class galois_insert_bag;
      explicit iterator(galois_insert_bag* P) :b(0), cThreadID(0), parent(P) {
	incr(); //move past initial null
      }

      void nextThread() {
	if (cThreadID < GaloisRuntime::getMaxThreadID()) {
	  ++cThreadID;
	  b = parent->getHeadFor(cThreadID);
	} else { //at end
	  parent = 0;
	}
      }

      void incr() { 
	if (!parent) return;
	if (b)
	  b = b->next; 
	while (!b && parent) //we didn't have one or we fell off
	  nextThread();
      }

    public:
      typedef ptrdiff_t                 difference_type;
      typedef std::forward_iterator_tag iterator_category;
      typedef T                         value_type;
      typedef T*                        pointer;
      typedef T&                        reference;

      iterator() :b(0), cThreadID(0),parent(0) {}

      bool operator==(const iterator& rhs) const { return b == rhs.b; }
      bool operator!=(const iterator& rhs) const { return b != rhs.b; }

      reference operator*()  const { assert(parent && b); return b->item; }
      pointer   operator->() const { return &(operator*()); }
      iterator& operator++() { incr(); return *this; }
      iterator  operator++(int) { iterator __tmp = *this; incr(); return __tmp; }
    };

    iterator begin() {
      return iterator(this);
    }

    iterator end() {
      return iterator();
    }

    //Only this is thread safe
    reference push(const T& val) {
      bag_item* B = new bag_item(val);
      i_push(B);
      return B->item;
    }
    
  };
}
#endif
