/** Alternate implementation of Bag -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
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
 * @author ahassaan@ices.utexas.edu
 */
#ifndef GALOIS_ALT_BAG_H
#define GALOIS_ALT_BAG_H

#include "Galois/BoundedVector.h"
#include "Galois/PerThreadContainer.h"
#include "Galois/DynamicArray.h"

#include "Galois/Runtime/Mem.h"

#include "Galois/Substrate/gio.h"


#include <boost/iterator/iterator_facade.hpp>

#include <list>



namespace Galois {

#define NEW_SERIAL_BAG_IMPL 1

#ifdef NEW_SERIAL_BAG_IMPL

template <typename T, const size_t SZ=0>
class SerialBag {

  using dbg = Substrate::debug<0>;

  using PageHeap = Runtime::PageHeap;
  using FSheap = Runtime::FixedSizeHeap;

  struct Block {
    Block* next;
    Block* prev;
    DynamicBoundedVector<T> chunk;

    explicit Block (T* beg=nullptr, T* end=nullptr)
      :
        next (nullptr),
        prev (nullptr),
        chunk (beg, end)
    {}
  };



  PageHeap* pageHeap;
  FSheap fsHeap;
  Block sentinel; // sentinel
  Block* tail;

  void init (void) {
    tail = &sentinel;
    sentinel.next = tail;
    sentinel.prev = tail;
  }

  //! sentinel is a dummy empty block that must be after tail
  void checkInvariants (void) const {
    assert (tail->next == &sentinel);
    assert (sentinel.prev == tail);
  }

  Block* getHead (void) const {
    return sentinel.next;
  }

  template <typename U>
  struct OuterIterImpl: public boost::iterator_facade<OuterIterImpl<U>, U, boost::bidirectional_traversal_tag> {

    friend class boost::iterator_core_access;

    Block* curr;

    explicit OuterIterImpl (Block* b=nullptr): curr (b) {}

    U& dereference (void)  const {
      assert (curr != nullptr);
      return curr->chunk;
    }

    // const U& dereference (void) const {
      // assert (curr != nullptr);
      // return curr->chunk;
    // }

    void increment (void) {
      curr = curr->next;
    }

    void decrement (void) {
      curr = curr->prev;
    }

    bool equal (const OuterIterImpl& that) const {
      return curr == that.curr;
    }

  };

  using OuterIter = OuterIterImpl<DynamicBoundedVector<T> >;

  OuterIter make_outer_beg (void) {
    return OuterIter (getHead ());
  }

  OuterIter make_outer_end (void) {
    return OuterIter (&sentinel);
  }

  using ConstOuterIter = OuterIterImpl<const DynamicBoundedVector<T> >;

  ConstOuterIter make_outer_cbeg (void) const {
    return ConstOuterIter (getHead ());
  }

  ConstOuterIter make_outer_cend (void) const {
    return ConstOuterIter (&sentinel);
  }

  using RevOuterIter = std::reverse_iterator<OuterIter>;

  RevOuterIter make_outer_rbeg (void) {
    return RevOuterIter (make_outer_end ());
  }

  RevOuterIter make_outer_rend (void) {
    return RevOuterIter (make_outer_beg ());
  }

  using ConstRevOuterIter = std::reverse_iterator<ConstOuterIter>;

  ConstRevOuterIter make_outer_crbeg (void) const {
    return ConstRevOuterIter (make_outer_cend ());
  }

  ConstRevOuterIter make_outer_crend (void) const {
    return ConstRevOuterIter (make_outer_cbeg ());
  }

  static const size_t ALLOC_SIZE = (SZ == 0) ? PageHeap::AllocSize : SZ * sizeof (T);
  static_assert (sizeof(T) < PageHeap::AllocSize, "Serial Bag with template type too large in size");
  static_assert (ALLOC_SIZE <= PageHeap::AllocSize, "Serial Bag with SZ parameter too large");
  static_assert (ALLOC_SIZE > (sizeof (T) + sizeof (Block)), "Serial Bag with SZ parameter too small");


  Block* newBlock (void) {

    void* p = nullptr;

    if (SZ == 0) { 
      assert (pageHeap);
      p = pageHeap->allocate (ALLOC_SIZE);

    } else {
      p = fsHeap.allocate (ALLOC_SIZE);
    }

    assert (p != nullptr);
    dbg::print (this, " allocating a new block: ", p);


    size_t offset = 1;
    if (sizeof (T) < sizeof (Block)) {
      offset += sizeof (Block) / sizeof (T);
    }

    T* beg = reinterpret_cast<T*> (p) + offset;
    T* end = reinterpret_cast<T*> (p) + (ALLOC_SIZE / sizeof (T));

    Block* b = reinterpret_cast<Block*> (p);
    ::new (b) Block (beg, end);

    return b;
  }

  void deleteBlock (Block* const b) {
    b->~Block ();

    if (SZ == 0) {
      assert (pageHeap);
      pageHeap->deallocate (b);
      dbg::print (this, " deallocating block: ", b);
    } else {
      fsHeap.deallocate (b);
    }
  };

  void pushBackBlock (void) {

    checkInvariants ();

    Block* b = newBlock ();

    b->next = &sentinel;
    sentinel.prev = b;
    b->prev = tail;
    tail->next = b;
    tail = b;

    checkInvariants ();
    // b->next = head;
    // head->prev = b;
    // head = b;
  }

  void popBackBlock (void) {

    assert (tail != nullptr && tail != &sentinel && !tail->chunk.empty ());
    assert (tail->next == &sentinel);

    Block* old_t = tail;

    Block* new_t = tail->prev;
    new_t->next = &sentinel;
    sentinel.prev = new_t;

    tail = new_t;

    deleteBlock (old_t);
    checkInvariants ();
  }

  void pushFrontBlock (void) {
    checkInvariants ();

    Block* b = newBlock ();


    Block* head = getHead ();
    b->next = head;
    head->prev = b;
    b->prev = &sentinel;
    sentinel.next = b;

    if (tail == &sentinel) {
      tail = b;
    }

    checkInvariants ();
  }


  void popFrontBlock (void) {
    checkInvariants ();

    Block* head = getHead ();

    assert (head != nullptr && head != &sentinel && head->chunk.empty ());
    assert (head->prev == &sentinel);

    sentinel.next = head->next;
    head->next->prev = &sentinel;

    if (tail == head) {
      tail = &sentinel;
      init ();
    }

    deleteBlock (head);
    checkInvariants ();
  }

  void printBlocks (void) {
    std::printf ("SerialBag blocks are: ");
    for (Block* i = getHead (); i != &sentinel; i = i->next) {
      auto& c = i->chunk;
      std::printf ("<curr:%p, next:%p, prev:%p, dbeg:%p, dsize:%p, dend:%p>, ",
          i, i->next, i->prev, c.begin (), c.begin () + c.size (), c.begin () + c.capacity ());
    }
    std::printf ("\n");
  }

public:
  using value_type = T;
  using reference = T&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using const_reference = const  value_type&;
  using pointer = value_type*;
  using const_pointer = const  value_type*;
   

  using iterator =  typename ChooseStlTwoLevelIterator<OuterIter, typename DynamicBoundedVector<T>::iterator>::type;
  using const_iterator =  typename ChooseStlTwoLevelIterator<ConstOuterIter, typename DynamicBoundedVector<T>::const_iterator>::type;

  using reverse_iterator =  typename ChooseStlTwoLevelIterator<RevOuterIter, typename DynamicBoundedVector<T>::reverse_iterator>::type;
  using const_reverse_iterator =  typename ChooseStlTwoLevelIterator<ConstRevOuterIter, typename DynamicBoundedVector<T>::const_reverse_iterator>::type;

  SerialBag (void)
    : 
      pageHeap (PageHeap::getInstance ()),
      fsHeap (ALLOC_SIZE),
      sentinel (),
      tail (&sentinel)
  {
    init ();
  }

  ~SerialBag (void) {
    clear ();
  }

  bool empty (void) const {
    return tail == &sentinel;
  }

  size_t size (void) const {

    size_t s = 0;
    for (Block* i = getHead (); i != &sentinel; i = i->next) {
      s += i->chunk.size ();
    }

    return s;
  }

  void clear (void) {

    for (Block* i = getHead (); i != &sentinel;) {
      Block* b = i;
      i = i->next;
      b->chunk.clear ();
      popFrontBlock ();
    }

    init ();
  }


  template <typename... Args>
  void emplace_back (Args&&... args) {

    if (empty () || tail->chunk.full ()) {
      pushBackBlock ();
      // printBlocks ();
    }

    tail->chunk.emplace_back (std::forward<Args> (args)...);
  }

  void push_back (const T& elem) {
    this->emplace_back (elem);
  }

  reference back (void) {
    assert (!empty ());
    return tail->chunk.back ();
  }

  const_reference back (void) const {
    return const_cast<SerialBag*> (this)->back ();
  }

  void pop_back (void) {
    assert (!empty ());

    tail->chunk.pop_back ();

    if (tail->chunk.empty ()) {
      popBackBlock ();
    }
  }

  void splice (SerialBag& that) {
    this->checkInvariants ();
    that.checkInvariants ();

    if (!that.empty ()) {
      this->tail->next = that.getHead ();
      that.getHead ()->prev = this->tail;

      this->tail = that.tail;
      this->tail->next = &(this->sentinel);
      this->sentinel.prev = this->tail;

      that.init (); // make that look empty 
    }

    assert (that.empty ());
    this->checkInvariants ();
    that.checkInvariants ();
  }

  iterator begin () {
    return stl_two_level_begin (make_outer_beg (), make_outer_end ());
  }

  iterator end () {
    return stl_two_level_end (make_outer_beg (), make_outer_end ());
  }

  const_iterator cbegin () const {
    return stl_two_level_cbegin (make_outer_cbeg (), make_outer_cend ());
  }

  const_iterator cend () const {
    return stl_two_level_cend (make_outer_cbeg (), make_outer_cend ());
  }

  const_iterator begin () const { 
    return cbegin ();
  }

  const_iterator end () const {
    return cend ();
  }

  reverse_iterator rbegin () {
    return stl_two_level_rbegin (make_outer_rbeg (), make_outer_rend ());
  }

  reverse_iterator rend () {
    return stl_two_level_rend (make_outer_rbeg (), make_outer_rend ());
  }

  const_reverse_iterator crbegin () {
    return stl_two_level_crbegin (make_outer_crbeg (), make_outer_crend ());
  }

  const_reverse_iterator crend () {
    return stl_two_level_crend (make_outer_crbeg (), make_outer_crend ());
  }

  const_reverse_iterator rbegin () const {
    return crbegin ();
  }

  const_reverse_iterator crend () const {
    return crend ();
  }

};

#else 
// 
// template <typename T, const size_t SZ=0>
// class SerialBag {
// 
//   using PageHeap = Runtime::MM::SystemHeap;
//   using FSheap = Runtime::MM::FixedSizeHeap;
// 
//   struct Block {
//     Block* next;
//     Block* prev;
//     DynamicBoundedVector<T> chunk;
// 
//     explicit Block (T* beg=nullptr, T* end=nullptr)
//       :
//         next (nullptr),
//         prev (nullptr),
//         chunk (beg, end)
//     {}
//   };
// 
// 
// 
//   PageHeap pageHeap;
//   FSheap fsHeap;
//   Block tail; // sentinel
//   Block* head;
// 
// 
//   template <typename U>
//   struct OuterIterImpl: public boost::iterator_facade<OuterIterImpl<U>, U, boost::bidirectional_traversal_tag> {
// 
//     friend class boost::iterator_core_access;
// 
//     Block* curr;
// 
//     explicit OuterIterImpl (Block* b=nullptr): curr (b) {}
// 
//     U& dereference (void)  const {
//       assert (curr != nullptr);
//       return curr->chunk;
//     }
// 
//     // const U& dereference (void) const {
//       // assert (curr != nullptr);
//       // return curr->chunk;
//     // }
// 
//     void increment (void) {
//       curr = curr->next;
//     }
// 
//     void decrement (void) {
//       curr = curr->prev;
//     }
// 
//     bool equal (const OuterIterImpl& that) const {
//       return curr == that.curr;
//     }
// 
//   };
// 
//   using OuterIter = OuterIterImpl<DynamicBoundedVector<T> >;
// 
//   OuterIter make_outer_beg (void) {
//     return OuterIter (head);
//   }
// 
//   OuterIter make_outer_end (void) {
//     return OuterIter (&tail);
//   }
// 
//   using ConstOuterIter = OuterIterImpl<const DynamicBoundedVector<T> >;
// 
//   ConstOuterIter make_outer_cbeg (void) const {
//     return ConstOuterIter (head);
//   }
// 
//   ConstOuterIter make_outer_cend (void) const {
//     return ConstOuterIter (&tail);
//   }
// 
//   using RevOuterIter = std::reverse_iterator<OuterIter>;
// 
//   RevOuterIter make_outer_rbeg (void) {
//     return RevOuterIter (make_outer_end ());
//   }
// 
//   RevOuterIter make_outer_rend (void) {
//     return RevOuterIter (make_outer_beg ());
//   }
// 
//   using ConstRevOuterIter = std::reverse_iterator<ConstOuterIter>;
// 
//   ConstRevOuterIter make_outer_crbeg (void) const {
//     return ConstRevOuterIter (make_outer_cend ());
//   }
// 
//   ConstRevOuterIter make_outer_crend (void) const {
//     return ConstRevOuterIter (make_outer_cbeg ());
//   }
// 
//   static const size_t ALLOC_SIZE = (SZ == 0) ? PageHeap::AllocSize : SZ * sizeof (T);
//   static_assert (sizeof(T) < PageHeap::AllocSize, "Serial Bag with template type too large in size");
//   static_assert (ALLOC_SIZE <= PageHeap::AllocSize, "Serial Bag with SZ parameter too large");
//   static_assert (ALLOC_SIZE > sizeof (Block), "Serial Bag with SZ parameter too small");
// 
// 
//   Block* allocateBlock (void) {
// 
//     void* p = nullptr;
// 
//     if (SZ == 0) { 
//       p = pageHeap.allocate (ALLOC_SIZE);
// 
//     } else {
//       p = fsHeap.allocate (ALLOC_SIZE);
//     }
// 
//     assert (p != nullptr);
// 
// 
//     size_t offset = 1;
//     if (sizeof (T) < sizeof (Block)) {
//       offset += sizeof (Block) / sizeof (T);
//     }
// 
//     T* beg = reinterpret_cast<T*> (p) + offset;
//     T* end = reinterpret_cast<T*> (p) + (ALLOC_SIZE / sizeof (T));
// 
//     Block* b = reinterpret_cast<Block*> (p);
//     ::new (b) Block (beg, end);
// 
//     return b;
//   }
// 
//   void pushBlock (void) {
// 
//     Block* b = allocateBlock ();
// 
//     b->next = head;
//     head->prev = b;
//     head = b;
//   }
// 
//   void popBlock (void) {
//     assert (head->chunk.empty ());
// 
//     Block* b = head;
//     head->next->prev = nullptr;
//     head = head->next;
// 
//     b->~Block ();
// 
//     if (SZ == 0) {
//       pageHeap.deallocate (b);
//     } else {
//       fsHeap.deallocate (b);
//     }
//   }
// 
// 
// 
// public:
//   using value_type = T;
//   using reference = T&;
//   using size_type = size_t;
//   using difference_type = ptrdiff_t;
//   using const_reference = const  value_type&;
//   using pointer = value_type*;
//   using const_pointer = const  value_type*;
//    
// 
//   using iterator =  typename ChooseStlTwoLevelIterator<OuterIter, typename DynamicBoundedVector<T>::iterator>::type;
//   using const_iterator =  typename ChooseStlTwoLevelIterator<ConstOuterIter, typename DynamicBoundedVector<T>::const_iterator>::type;
// 
//   using reverse_iterator =  typename ChooseStlTwoLevelIterator<RevOuterIter, typename DynamicBoundedVector<T>::reverse_iterator>::type;
//   using const_reverse_iterator =  typename ChooseStlTwoLevelIterator<ConstRevOuterIter, typename DynamicBoundedVector<T>::const_reverse_iterator>::type;
// 
//   SerialBag (void)
//     : 
//       pageHeap (),
//       fsHeap (ALLOC_SIZE),
//       tail (),
//       head (&tail)
//   {}
// 
//   ~SerialBag (void) {
//     clear ();
//   }
// 
//   bool empty (void) const {
//     return head == &tail;
//   }
// 
//   size_t size (void) const {
// 
//     size_t s = 0;
//     for (Block* i = head; i != &tail; i = i->next) {
//       s += i->chunk.size ();
//     }
// 
//     return s;
//   }
// 
//   void clear (void) {
// 
//     for (Block* i = head; i != &tail;) {
//       Block* b = i;
//       i = i->next;
//       b->chunk.clear ();
//       popBlock ();
//     }
// 
//     head = &tail;
//   }
// 
//   void printBlocks (void) {
//     std::printf ("SerialBag blocks are: ");
//     for (Block* i = head; i != &tail; i = i->next) {
//       auto& c = i->chunk;
//       std::printf ("<curr:%p, next:%p, prev:%p, dbeg:%p, dsize:%p, dend:%p>, ",
//           i, i->next, i->prev, c.begin (), c.begin () + c.size (), c.begin () + c.capacity ());
//     }
//     std::printf ("\n");
//   }
// 
// 
//   template <typename... Args>
//   void emplace_back (Args&&... args) {
// 
//     if (empty () || head->chunk.full ()) {
//       pushBlock ();
//       // printBlocks ();
//     }
// 
//     head->chunk.emplace_back (std::forward<Args> (args)...);
//   }
// 
//   void push_back (const T& elem) {
//     this->emplace_back (elem);
//   }
// 
//   reference back (void) {
//     assert (!empty ());
//     return head->chunk.back ();
//   }
// 
//   const_reference back (void) const {
//     return const_cast<SerialBag*> (this)->back ();
//   }
// 
//   void pop_back (void) {
//     assert (!empty ());
// 
//     head->chunk.pop_back ();
// 
//     if (head->chunk.empty ()) {
//       popBlock ();
//     }
//   }
//   iterator begin () {
//     return stl_two_level_begin (make_outer_beg (), make_outer_end ());
//   }
// 
//   iterator end () {
//     return stl_two_level_end (make_outer_beg (), make_outer_end ());
//   }
// 
//   const_iterator cbegin () const {
//     return stl_two_level_cbegin (make_outer_cbeg (), make_outer_cend ());
//   }
// 
//   const_iterator cend () const {
//     return stl_two_level_cend (make_outer_cbeg (), make_outer_cend ());
//   }
// 
//   const_iterator begin () const { 
//     return cbegin ();
//   }
// 
//   const_iterator end () const {
//     return cend ();
//   }
// 
//   reverse_iterator rbegin () {
//     return stl_two_level_rbegin (make_outer_rbeg (), make_outer_rend ());
//   }
// 
//   reverse_iterator rend () {
//     return stl_two_level_rend (make_outer_rbeg (), make_outer_rend ());
//   }
// 
//   const_reverse_iterator crbegin () {
//     return stl_two_level_crbegin (make_outer_crbeg (), make_outer_crend ());
//   }
// 
//   const_reverse_iterator crend () {
//     return stl_two_level_crend (make_outer_crbeg (), make_outer_crend ());
//   }
// 
//   const_reverse_iterator rbegin () const {
//     return crbegin ();
//   }
// 
//   const_reverse_iterator crend () const {
//     return crend ();
//   }
// 
// };
// 
#endif

template <typename T, const size_t SZ=0>
class PerThreadBag: public PerThreadContainer<SerialBag<T, SZ> > {
  using C = SerialBag<T, SZ>;
  using Super_ty = PerThreadContainer<C>;

public:

  PerThreadBag (): Super_ty () {
    Super_ty::init ();
  }

  void push_back (const T& x) {
    Super_ty::get ().push_back (x);
  }

  void push (const T& x) {
    push_back (x);
  }

  void splice_all (PerThreadBag& that) {
    for (unsigned i = 0; i < Super_ty::numRows(); ++i) {
      Super_ty::get(i).splice (that.Super_ty::get (i));
    }
  }
};

} // end namespace Galois

#endif // GALOIS_ALT_BAG_H
