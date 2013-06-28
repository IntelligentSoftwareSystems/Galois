/** (d)Chunked(F|L)ifo worklist -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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

#ifndef GALOIS_WORKLIST_CHUNKED_H
#define GALOIS_WORKLIST_CHUNKED_H

#include "Galois/FixedSizeRing.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/WorkList/WorkListHelpers.h"
#include "WLCompileCheck.h"

namespace Galois {
namespace WorkList {

//This overly complex specialization avoids a pointer indirection for non-distributed WL when accessing PerLevel
template<bool d, typename TQ>
struct squeues;

template<typename TQ>
struct squeues<true,TQ> {
  Runtime::PerPackageStorage<TQ> queues;
  TQ& get(int i) { return *queues.getRemote(i); }
  TQ& get() { return *queues.getLocal(); }
  int myEffectiveID() { return Runtime::LL::getTID(); }
  int size() { return Runtime::activeThreads; }
};

template<typename TQ>
struct squeues<false,TQ> {
  TQ queue;
  TQ& get(int i) { return queue; }
  TQ& get() { return queue; }
  int myEffectiveID() { return 0; }
  int size() { return 0; }
};

//! Common functionality to all chunked worklists
template<typename T, template<typename, bool> class QT, bool Distributed, bool IsStack, int ChunkSize, bool Concurrent>
struct ChunkedMaster : private boost::noncopyable {
  template<bool _concurrent>
  using rethread = ChunkedMaster<T, QT, Distributed, IsStack, ChunkSize, _concurrent>;

  template<typename _T>
  using retype = ChunkedMaster<_T, QT, Distributed, IsStack, ChunkSize, Concurrent>;

  template<int _chunk_size>
  using with_chunk_size = ChunkedMaster<T, QT, Distributed, IsStack, _chunk_size, Concurrent>;


private:
  class Chunk : public FixedSizeRing<T, ChunkSize>, public QT<Chunk, Concurrent>::ListNode {};

  Runtime::MM::FixedSizeAllocator heap;

  struct p {
    Chunk* cur;
    Chunk* next;
  };

  typedef QT<Chunk, Concurrent> LevelItem;

  Runtime::PerThreadStorage<p> data;
  squeues<Distributed, LevelItem> Q;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
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

  T* pushi(const T& val, p* n)  {
    T* retval = 0;

    if (n->next && (retval = n->next->push_back(val)))
      return retval;
    if (n->next)
      pushChunk(n->next);
    n->next = mkChunk();
    retval = n->next->push_back(val);
    assert(retval);
    return retval;
  }

public:
  typedef T value_type;

  ChunkedMaster() : heap(sizeof(Chunk)) { }

  void flush() {
    p& n = *data.getLocal();
    if (n.next)
      pushChunk(n.next);
    n.next = 0;
  }
  
  //! Most worklists have void return value for push. This push returns address
  //! of placed item to facilitate some internal runtime uses. The address is
  //! generally not safe to use in the presence of concurrent pops.
  value_type* push(const value_type& val)  {
    p* n = data.getLocal();
    return pushi(val, n);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    p* n = data.getLocal();
    while (b != e)
      pushi(*b++, n);
  }

  template<typename RangeTy>
  void push_initial(const RangeTy& range) {
    auto rp = range.local_pair();
    push(rp.first, rp.second);
  }

  boost::optional<value_type> pop()  {
    p& n = *data.getLocal();
    boost::optional<value_type> retval;
    if (IsStack) {
      if (n.next && (retval = n.next->extract_back()))
	return retval;
      if (n.next)
	delChunk(n.next);
      n.next = popChunk();
      if (n.next)
	return n.next->extract_back();
      return boost::optional<value_type>();
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
      return boost::optional<value_type>();
    }
  }
};

/**
 * Chunked FIFO. A global FIFO of chunks of some fixed size.
 *
 * @tparam ChunkSize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
class ChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, false, false, ChunkSize, Concurrent> {};
GALOIS_WLCOMPILECHECK(ChunkedFIFO)

/**
 * Chunked LIFO. A global LIFO of chunks of some fixed size.
 *
 * @tparam ChunkSize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
class ChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, false, true, ChunkSize, Concurrent> {};
GALOIS_WLCOMPILECHECK(ChunkedLIFO)

/**
 * Distributed chunked FIFO. A more scalable version of {@link ChunkedFIFO}.
 *
 * @tparam ChunkSize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
class dChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, true, false, ChunkSize, Concurrent> {};
GALOIS_WLCOMPILECHECK(dChunkedFIFO)

/**
 * Distributed chunked LIFO. A more scalable version of {@link ChunkedLIFO}.
 *
 * @tparam chunksize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
class dChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, true, true, ChunkSize, Concurrent> {};
GALOIS_WLCOMPILECHECK(dChunkedLIFO)

/**
 * Distributed chunked bag. A scalable and resource-efficient policy when you
 * are agnostic to the particular scheduling order.
 *
 * @tparam chunksize chunk size
 */
template<int ChunkSize=64, typename T = int, bool Concurrent=true>
class dChunkedBag : public ChunkedMaster<T, ConExtLinkedQueue, true, true, ChunkSize, Concurrent> {};
GALOIS_WLCOMPILECHECK(dChunkedBag)


} // end namespace WorkList
} // end namespace Galois

#endif
