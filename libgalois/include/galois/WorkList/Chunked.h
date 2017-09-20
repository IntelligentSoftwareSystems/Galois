/** (d)Chunked(F|L)ifo worklist -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_WORKLIST_CHUNKED_H
#define GALOIS_WORKLIST_CHUNKED_H

#include "galois/FixedSizeRing.h"
#include "galois/Substrate/PaddedLock.h"
#include "galois/runtime/Mem.h"
#include "galois/WorkList/WorkListHelpers.h"
#include "WLCompileCheck.h"

namespace galois {
namespace runtime {
extern unsigned activeThreads;
}
namespace worklists {

namespace detail {
//This overly complex specialization avoids a pointer indirection for non-distributed WL when accessing PerLevel
template<bool, template<typename> class PS, typename TQ>
struct squeue {
  PS<TQ> queues;
  TQ& get(int i) { return *queues.getRemote(i); }
  TQ& get() { return *queues.getLocal(); }
  int myEffectiveID() { return substrate::ThreadPool::getTID(); }
  int size() { return runtime::activeThreads; }
};

template<template<typename> class PS, typename TQ>
struct squeue<false, PS, TQ> {
  TQ queue;
  TQ& get(int i) { return queue; }
  TQ& get() { return queue; }
  int myEffectiveID() { return 0; }
  int size() { return 0; }
};

//! Common functionality to all chunked worklists
template<typename T, template<typename, bool> class QT, bool Distributed, bool IsStack, int ChunkSize, bool Concurrent>
struct ChunkedMaster : private boost::noncopyable {
  template<typename _T>
  using retype = ChunkedMaster<_T, QT, Distributed, IsStack, ChunkSize, Concurrent>;

  template<int _chunk_size>
  using with_chunk_size = ChunkedMaster<T, QT, Distributed, IsStack, _chunk_size, Concurrent>;

  template<bool _Concurrent>
  using rethread = ChunkedMaster<T, QT, Distributed, IsStack, ChunkSize, _Concurrent>;

private:
  class Chunk : public FixedSizeRing<T, ChunkSize>, public QT<Chunk, Concurrent>::ListNode {};

  runtime::FixedSizeAllocator<Chunk> alloc;

  struct p {
    Chunk* cur;
    Chunk* next;
    p(): cur(0), next(0) { }
  };

  typedef QT<Chunk, Concurrent> LevelItem;

  squeue<Concurrent, substrate::PerThreadStorage, p> data;
  squeue<Distributed, substrate::PerPackageStorage, LevelItem> Q;

  Chunk* mkChunk() {
    Chunk* ptr = alloc.allocate(1);
    alloc.construct(ptr);
    return ptr;
  }
  
  void delChunk(Chunk* ptr) {
    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
  }

  void pushChunk(Chunk* C)  {
    LevelItem& I = Q.get();
    I.push(C);
  }

  Chunk* popChunkByID(unsigned int i)  {
    LevelItem& I = Q.get(i);
    return I.pop();
  }

  Chunk* popChunk()  {
    int id = Q.myEffectiveID();
    Chunk* r = popChunkByID(id);
    if (r)
      return r;

    for (int i = id + 1; i < (int) Q.size(); ++i) {
      r = popChunkByID(i);
      if (r) 
	return r;
    }

    for (int i = 0; i < id; ++i) {
      r = popChunkByID(i);
      if (r)
	return r;
    }

    return 0;
  }

  template<typename... Args>
  T* emplacei(p& n, Args&&... args)  {
    T* retval = 0;
    if (n.next && (retval = n.next->emplace_back(std::forward<Args>(args)...)))
      return retval;
    if (n.next)
      pushChunk(n.next);
    n.next = mkChunk();
    retval = n.next->emplace_back(std::forward<Args>(args)...);
    assert(retval);
    return retval;
  }

public:
  typedef T value_type;

  ChunkedMaster() { }

  void flush() {
    p& n = data.get();
    if (n.next)
      pushChunk(n.next);
    n.next = 0;
  }
  
  /**
   * Construct an item on the worklist and return a pointer to its value.
   *
   * This pointer facilitates some internal runtime uses and is not designed
   * to be used by general clients. The address is generally not safe to use
   * in the presence of concurrent pops.
   */
  template<typename... Args>
  value_type* emplace(Args&&... args) {
    p& n = data.get();
    return emplacei(n, std::forward<Args>(args)...);
  }

  /**
   * Return pointer to next value to be returned by pop.
   *
   * For internal runtime use.
   */
  value_type* peek() {
    p& n = data.get();
    if (IsStack) {
      if (n.next && !n.next->empty())
	return &n.next->back();
      if (n.next)
	delChunk(n.next);
      n.next = popChunk();
      if (n.next && !n.next->empty())
	return &n.next->back();
      return NULL;
    } else {
      if (n.cur && !n.cur->empty())
	return &n.cur->front();
      if (n.cur)
	delChunk(n.cur);
      n.cur = popChunk();
      if (!n.cur) {
	n.cur = n.next;
	n.next = 0;
      }
      if (n.cur && !n.cur->empty())
	return &n.cur->front();
      return NULL;
    }
  }

  /**
   * Remove the value returned from peek() from the worklist. 
   *
   * For internal runtime use.
   */
  void pop_peeked() {
    p& n = data.get();
    if (IsStack) {
      n.next->pop_back();
      return;
    } else {
      n.cur->pop_front();
      return;
    }
  }

  void push(const value_type& val)  {
    p& n = data.get();
    emplacei(n, val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    p& n = data.get();
    while (b != e)
      emplacei(n, *b++);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  galois::optional<value_type> pop()  {
    p& n = data.get();
    galois::optional<value_type> retval;
    if (IsStack) {
      if (n.next && (retval = n.next->extract_back()))
	return retval;
      if (n.next)
	delChunk(n.next);
      n.next = popChunk();
      if (n.next)
	return n.next->extract_back();
      return galois::optional<value_type>();
    } else {
      if (n.cur && (retval = n.cur->extract_front()))
	return retval;
      if (n.cur)
	delChunk(n.cur);
      n.cur = popChunk();
      if (!n.cur) {
	n.cur = n.next;
	n.next = 0;
      }
      if (n.cur)
	return n.cur->extract_front();
      return galois::optional<value_type>();
    }
  }
};

} // namespace detail

/**
 * Chunked FIFO. A global FIFO of chunks of some fixed size.
 *
 * @tparam ChunkSize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
using ChunkedFIFO = detail::ChunkedMaster<T, ConExtLinkedQueue, false, false, ChunkSize, Concurrent>;
GALOIS_WLCOMPILECHECK(ChunkedFIFO)

/**
 * Chunked LIFO. A global LIFO of chunks of some fixed size.
 *
 * @tparam ChunkSize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
using ChunkedLIFO = detail::ChunkedMaster<T, ConExtLinkedStack, false, true, ChunkSize, Concurrent>;
GALOIS_WLCOMPILECHECK(ChunkedLIFO)

/**
 * Distributed chunked FIFO. A more scalable version of {@link ChunkedFIFO}.
 *
 * @tparam ChunkSize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
using dChunkedFIFO = detail::ChunkedMaster<T, ConExtLinkedQueue, true, false, ChunkSize, Concurrent>;
GALOIS_WLCOMPILECHECK(dChunkedFIFO)

/**
 * Distributed chunked LIFO. A more scalable version of {@link ChunkedLIFO}.
 *
 * @tparam chunksize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
using dChunkedLIFO = detail::ChunkedMaster<T, ConExtLinkedStack, true, true, ChunkSize, Concurrent>;
GALOIS_WLCOMPILECHECK(dChunkedLIFO)

/**
 * Distributed chunked bag. A scalable and resource-efficient policy when you
 * are agnostic to the particular scheduling order.
 *
 * @tparam chunksize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
using dChunkedBag = detail::ChunkedMaster<T, ConExtLinkedQueue, true, true, ChunkSize, Concurrent>;
GALOIS_WLCOMPILECHECK(dChunkedBag)


} // end namespace worklists
} // end namespace galois

#endif
