/** (d)Chunked(F|L)ifo worklist -*- C++ -*-
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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_WORKLIST_CHUNKED_H
#define GALOIS_WORKLIST_CHUNKED_H

#include "Galois/Runtime/ll/PaddedLock.h"
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

template<typename T, template<typename, bool> class QT, bool distributed = false, bool isStack = false, int chunksize=64, bool concurrent=true>
class ChunkedMaster : private boost::noncopyable {
  class Chunk : public Galois::FixedSizeRing<T, chunksize>, public QT<Chunk, concurrent>::ListNode {};

  Runtime::MM::FixedSizeAllocator heap;

  struct p {
    Chunk* cur;
    Chunk* next;
  };

  typedef QT<Chunk, concurrent> LevelItem;

  Runtime::PerThreadStorage<p> data;
  squeues<distributed, LevelItem> Q;

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

  template<bool newconcurrent>
  using rethread = ChunkedMaster<T, QT, distributed, isStack, chunksize, newconcurrent>;
  template<typename Tnew>
  using retype = ChunkedMaster<Tnew, QT, distributed, isStack, chunksize, concurrent>;

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
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  boost::optional<value_type> pop()  {
    p& n = *data.getLocal();
    boost::optional<value_type> retval;
    if (isStack) {
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

template<int chunksize=64, typename T = int, bool concurrent=true>
class ChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, false, false, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(ChunkedFIFO)

template<int chunksize=64, typename T = int, bool concurrent=true>
class ChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, false, true, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(ChunkedLIFO)

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedFIFO : public ChunkedMaster<T, ConExtLinkedQueue, true, false, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(dChunkedFIFO)

template<int chunksize=64, typename T = int, bool concurrent=true>
class dChunkedLIFO : public ChunkedMaster<T, ConExtLinkedStack, true, true, chunksize, concurrent> {};
GALOIS_WLCOMPILECHECK(dChunkedLIFO)


} // end namespace WorkList
} // end namespace Galois

#endif
