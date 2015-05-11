/** Alternate implementation of Bag -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * @author ahassaan@ices.utexas.edu
 */
#ifndef GALOIS_ALT_BAG_H
#define GALOIS_ALT_BAG_H

#include "Galois/BoundedVector.h"
#include "Galois/PerThreadContainer.h"
#include "Galois/DynamicArray.h"

#include "Galois/Runtime/mm/Mem.h"

#include <boost/iterator/iterator_facade.hpp>

#include <list>



namespace Galois {

#define NEW_SERIAL_BAG_IMPL 1

#ifdef NEW_SERIAL_BAG_IMPL

template <typename T, const size_t SZ=0>
class SerialBag {

  using PageHeap = Runtime::MM::SystemHeap;
  using FSheap = Runtime::MM::FixedSizeHeap;

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



  PageHeap pageHeap;
  FSheap fsHeap;
  Block tail; // sentinel
  Block* head;


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
    return OuterIter (head);
  }

  OuterIter make_outer_end (void) {
    return OuterIter (&tail);
  }

  using ConstOuterIter = OuterIterImpl<const DynamicBoundedVector<T> >;

  ConstOuterIter make_outer_cbeg (void) const {
    return ConstOuterIter (head);
  }

  ConstOuterIter make_outer_cend (void) const {
    return ConstOuterIter (&tail);
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
  static_assert (ALLOC_SIZE > sizeof (Block), "Serial Bag with SZ parameter too small");


  Block* allocateBlock (void) {

    void* p = nullptr;

    if (SZ == 0) { 
      p = pageHeap.allocate (ALLOC_SIZE);

    } else {
      p = fsHeap.allocate (ALLOC_SIZE);
    }

    assert (p != nullptr);


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

  void pushBlock (void) {

    Block* b = allocateBlock ();

    b->next = head;
    head->prev = b;
    head = b;
  }

  void popBlock (void) {
    assert (head->chunk.empty ());

    Block* b = head;
    head->next->prev = nullptr;
    head = head->next;

    b->~Block ();

    if (SZ == 0) {
      pageHeap.deallocate (b);
    } else {
      fsHeap.deallocate (b);
    }
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
      pageHeap (),
      fsHeap (ALLOC_SIZE),
      tail (),
      head (&tail)
  {}

  ~SerialBag (void) {
    clear ();
  }

  bool empty (void) const {
    return head == &tail;
  }

  size_t size (void) const {

    size_t s = 0;
    for (Block* i = head; i != &tail; i = i->next) {
      s += i->chunk.size ();
    }

    return s;
  }

  void clear (void) {

    for (Block* i = head; i != &tail;) {
      Block* b = i;
      i = i->next;
      b->chunk.clear ();
      popBlock ();
    }

    head = &tail;
  }

  void printBlocks (void) {
    std::printf ("SerialBag blocks are: ");
    for (Block* i = head; i != &tail; i = i->next) {
      auto& c = i->chunk;
      std::printf ("<curr:%p, next:%p, prev:%p, dbeg:%p, dsize:%p, dend:%p>, ",
          i, i->next, i->prev, c.begin (), c.begin () + c.size (), c.begin () + c.capacity ());
    }
    std::printf ("\n");
  }


  template <typename... Args>
  void emplace_back (Args&&... args) {

    if (empty () || head->chunk.full ()) {
      pushBlock ();
      // printBlocks ();
    }

    head->chunk.emplace_back (std::forward<Args> (args)...);
  }

  void push_back (const T& elem) {
    this->emplace_back (elem);
  }

  reference back (void) {
    assert (!empty ());
    return head->chunk.back ();
  }

  const_reference back (void) const {
    return const_cast<SerialBag*> (this)->back ();
  }

  void pop_back (void) {
    assert (!empty ());

    head->chunk.pop_back ();

    if (head->chunk.empty ()) {
      popBlock ();
    }
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

template <typename T, const size_t SZ=16*1024>
class SerialBag {
protected:


  using Chunk = BoundedVector<T, SZ>;
  using OuterList = std::list<Chunk, Runtime::MM::FixedSizeAllocator<Chunk> >;
  // using OuterList = typename ContainersWithGAlloc::Deque<Chunk>::type;


  OuterList m_outerList;
  Chunk* m_lastChunk;
  size_t m_size;

  void addLastChunk (void) {
    assert (m_lastChunk == getLastChunk ());
    assert (m_lastChunk == nullptr || m_lastChunk->full ());

    m_outerList.emplace_back ();
    m_lastChunk = &(m_outerList.back ());
  }



  Chunk* getLastChunk (void) {

    if (m_outerList.empty ()) {
      return nullptr;

    } else {
      Chunk& chunk = m_outerList.back ();
      assert (!chunk.empty ());
      return &chunk;
    }
  }
  
  Chunk* getFirstChunk (void) {

    if (m_outerList.empty ()) {
      return nullptr;

    } else {
      Chunk& chunk = m_outerList.front ();
      assert (!chunk.empty ());
      return &chunk;
    }
  }


public:

  using value_type = T;
  using reference = T&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using const_reference = const  value_type&;
  using pointer = value_type*;
  using const_pointer = const  value_type*;
   

  using iterator =  decltype(stl_two_level_begin (m_outerList.begin (), m_outerList.end ()));
  using const_iterator =  decltype(stl_two_level_cbegin (m_outerList.cbegin (), m_outerList.cend ()));
  using reverse_iterator =  decltype(stl_two_level_rbegin (m_outerList.rbegin (), m_outerList.rend ()));
  using const_reverse_iterator =  decltype(stl_two_level_crbegin (m_outerList.crbegin (), m_outerList.crend ()));

  SerialBag (): m_outerList (), m_size (0) {
  }

  ~SerialBag (void) {
    clear ();
    m_outerList.clear ();
  }

  size_t size () const { return m_size; }

  bool empty () const {
    auto b = m_outerList.begin ();
    auto e = m_outerList.end ();

    if (b == e) { 
      return true; 

    } else {
      // XXX: works if push_front and pop_front are
      // not supported
      return m_outerList.front ().empty ();

    }


    // ++b;
    // // m_outerList.front () empty () && m_outerList.size () == 1
    // return (m_outerList.front ().empty () && b == e);
  }

  template <typename... Args>
  void emplace_back (Args&&... args) {

    if (m_lastChunk != nullptr && !m_lastChunk->full ()) {
      m_lastChunk->emplace_back (std::forward<Args> (args)...);
      ++m_size;
      return;

    } else {

      addLastChunk ();
      emplace_back (std::forward<Args> (args)...);
    }
    
    // Chunk* chunk = getLastChunk ();
// 
    // if (chunk == nullptr || chunk->full ()) {
      // assert (m_outerList.empty () || chunk->full ());
      // m_outerList.emplace_back ();
      // chunk = &m_outerList.back ();
    // }
// 
    // chunk->emplace_back (std::forward<Args> (args)...);
    // ++m_size;
// 

  }

  void push_back (const T& elem) {
    this->emplace_back (elem);
  }

  //! error to call when empty
  //! Implementation does not check for empty container
  //! to keep logic simpler
  void pop_back (void) {

    // Chunk* chunk = getLastChunk ();
    // assert (chunk != nullptr);

    assert (m_lastChunk != nullptr);
    m_lastChunk->pop_back ();

    if (m_lastChunk->empty ()) {
      m_outerList.pop_back ();
      m_lastChunk = getLastChunk ();
    }

    --m_size;

    // Chunk& chunk = getLastChunk ();
// 
    // chunk.pop_back ();
// 
    // if(chunk.empty ()) {
      // m_outerList.pop_back ();
    // }
// 
    // if (m_outerList.empty ()) {
      // // restore the invariant of m_outerList containing at least one empty chunk
      // m_outerList.emplace_back ();
    // }
    // --m_size;
  }

  //! error to call when empty
  //! Implementation does not check for empty container
  //! to keep logic simpler
  reference back (void) {
    // Chunk* chunk = getLastChunk ();
    assert (!m_outerList.empty ());
    assert (m_lastChunk && m_lastChunk == getLastChunk ());
    return m_lastChunk->back ();
  }

  //! error to call when empty
  //! Implementation does not check for empty container
  //! to keep logic simpler
  const_reference back (void) const {
    return const_cast<SerialBag*> (this)->back ();
  }

  //! error to call when empty
  //! Implementation does not check for empty container
  //! to keep logic simpler
  reference front (void) {
    Chunk* chunk = getFirstChunk ();
    assert (chunk != nullptr);
    return chunk->front ();
  }

  //! error to call when empty
  //! Implementation does not check for empty container
  //! to keep logic simpler
  const_reference front (void) const {
    return const_cast<SerialBag*> (this)->front ();
  }

  void clear (void) {
    while (!m_outerList.empty ()) {
      m_outerList.pop_back ();
    }
    m_size = 0;
    m_lastChunk = nullptr;
  }

  void splice (SerialBag& that) {

    // m_size += that.m_size;
    // m_outerList.splice (m_outerList.end (), that.m_outerList);
    // assert (that.m_outerList.empty ());
    // that.clear ();

  }


  iterator begin () {
    return stl_two_level_begin (m_outerList.begin (), m_outerList.end ());
  }

  iterator end () {
    return stl_two_level_end (m_outerList.begin (), m_outerList.end ());
  }

  const_iterator cbegin () const {
    return stl_two_level_cbegin (m_outerList.begin (), m_outerList.end ());
  }

  const_iterator cend () const {
    return stl_two_level_cend (m_outerList.begin (), m_outerList.end ());
  }

  const_iterator begin () const { 
    return cbegin ();
  }

  const_iterator end () const {
    return cend ();
  }

};

#endif // NEW_SERIAL_BAG_IMPL

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

  // void splice_all (PerThreadBag& that) {
    // for (unsigned i = 0; i < Super_ty::numRows(); ++i) {
      // Super_ty::get(i).splice (that.Super_ty::get (i));
    // }
  // }
};

} // end namespace Galois

#endif // GALOIS_ALT_BAG_H
