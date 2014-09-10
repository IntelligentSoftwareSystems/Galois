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
#include "Galois/Runtime/mm/Mem.h"

#include <list>



namespace Galois {

template <typename T, const size_t SZ=16*1024>
class SerialBag {
protected:
  using Chunk = BoundedVector<T, SZ>;
  using OuterList = std::list<Chunk, Runtime::MM::FixedSizeAllocator<Chunk> >;


  OuterList outerList;
  size_t m_size;

  Chunk& getLastChunk (void) {
    assert (!outerList.empty ());
    
    Chunk& chunk = outerList.back ();

    assert (!chunk.empty ());

    return chunk;
  }
  
  Chunk& getFirstChunk (void) {
    assert (!outerList.empty ());

    Chunk& chunk = outerList.front ();

    assert (!chunk.empty ());

    return chunk;
  }


public:

  using value_type = T;
  using reference = T&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using const_reference = const  value_type&;
  using pointer = value_type*;
  using const_pointer = const  value_type*;
   

  using iterator =  decltype(stl_two_level_begin (outerList.begin (), outerList.end ()));
  using const_iterator =  decltype(stl_two_level_cbegin (outerList.cbegin (), outerList.cend ()));
  using reverse_iterator =  decltype(stl_two_level_rbegin (outerList.rbegin (), outerList.rend ()));
  using const_reverse_iterator =  decltype(stl_two_level_crbegin (outerList.crbegin (), outerList.crend ()));

  SerialBag (): outerList (), m_size (0) {
    outerList.emplace_back (); // put in 1 chunk to begin with
  }

  ~SerialBag (void) {
    clear ();
    outerList.clear ();
  }

  size_t size () const { return m_size; }

  bool empty () const {
    auto b = outerList.begin ();
    auto e = outerList.end ();

    assert (b != e); // outerList.size () >= 1;

    ++b;

    // outerList.front () empty () && outerList.size () == 1
    return (outerList.front ().empty () && b == e);
  }

  template <typename... Args>
  void emplace_back (Args&&... args) {
    assert (!outerList.empty ());

    Chunk* chunk = &(outerList.back ());

    if (chunk->full ()) {
      outerList.emplace_back ();
      chunk = &(outerList.back ());
    } 

    chunk->emplace_back (std::forward<Args> (args)...);
    ++m_size;
  }

  void push_back (const T& elem) {
    this->emplace_back (elem);
  }

  void pop_back (void) {
    Chunk& chunk = getLastChunk ();

    chunk.pop_back ();

    if(chunk.empty ()) {
      outerList.pop_back ();
    }

    if (outerList.empty ()) {
      // restore the invariant of outerList containing at least one empty chunk
      outerList.emplace_back ();
    }
    --m_size;
  }

  reference back (void) {
    Chunk& chunk = getLastChunk ();
    return chunk.back ();
  }

  const_reference back (void) const {
    return const_cast<SerialBag*> (this)->back ();
  }

  reference front (void) {
    Chunk& chunk = getFirstChunk ();
    return chunk.front ();
  }

  const_reference front (void) const {
    return const_cast<SerialBag*> (this)->front ();
  }

  void clear (void) {
    while (!outerList.empty ()) {
      outerList.pop_back ();
    }
    m_size = 0;

    outerList.emplace_back ();
  }

  iterator begin () {
    return stl_two_level_begin (outerList.begin (), outerList.end ());
  }

  iterator end () {
    return stl_two_level_end (outerList.begin (), outerList.end ());
  }

  const_iterator cbegin () const {
    return stl_two_level_cbegin (outerList.begin (), outerList.end ());
  }

  const_iterator cend () const {
    return stl_two_level_cend (outerList.begin (), outerList.end ());
  }

  const_iterator begin () const { 
    return cbegin ();
  }

  const_iterator end () const {
    return cend ();
  }
};

template <typename T, const size_t SZ=16*1024>
class PerThreadBag: public Runtime::PerThreadContainer<SerialBag<T, SZ> > {
  using C = SerialBag<T, SZ>;
  using Super_ty = Runtime::PerThreadContainer<C>;

public:

  PerThreadBag (): Super_ty () {
    Super_ty::init (C());
  }

  void push_back (const T& x) {
    Super_ty::get ().push_back (x);
  }

  void push (const T& x) {
    push_back (x);
  }
};

} // end namespace Galois

#endif // GALOIS_ALT_BAG_H
