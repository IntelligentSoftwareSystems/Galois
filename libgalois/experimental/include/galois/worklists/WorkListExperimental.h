/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_RUNTIME_WORKLISTEXPERIMENTAL_H
#define GALOIS_RUNTIME_WORKLISTEXPERIMENTAL_H

#include "galois/Bag.h"
#include "galois/optional.h"
#include "galois/Queue.h"

#include "galois/worklists/WorkList.h"
#include "galois/worklists/WorkListDebug.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/Termination.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/PaddedLock.h"

#include "llvm/Support/CommandLine.h"

#ifdef USE_TBB
#define TBB_PREVIEW_CONCURRENT_PRIORITY_QUEUE 1
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_priority_queue.h>
#include <tbb/concurrent_queue.h>
#endif

#include <boost/utility.hpp>
#include <cstdlib>
#include <set>

namespace galois {
namespace worklists {

template<class T, class Indexer = DummyIndexer<T>, typename ContainerTy = GFIFO<T>, bool concurrent=true >
class ApproxOrderByIntegerMetric : private boost::noncopyable {
  typename ContainerTy::template rethread<concurrent> data[2048];

  Indexer I;
  substrate::PerThreadStorage<unsigned int> cursor;

  int num() {
    return 2048;
  }

public:
  typedef T value_type;
  template<bool newconcurrent>
  struct rethread {
    typedef ApproxOrderByIntegerMetric<T, Indexer, ContainerTy, newconcurrent> type;
  };

  ApproxOrderByIntegerMetric(const Indexer& x = Indexer())
    :I(x)
  {
    for (int i = 0; i < cursor.size(); ++i)
      *cursor.getRemote(i) = 0;
  }

  void push(value_type val) {
    unsigned int index = I(val);
    index %= num();
    assert(index < num());
    return data[index].push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    unsigned int& cur = concurrent ? *cursor.getLocal() : *cursor.getRemote(0);
    galois::optional<value_type> ret = data[cur].pop();
    if (ret)
      return ret;

    //must move cursor
    for (int i = 0; i <= num(); ++i) {
      cur = (cur + 1) % num();
      ret = data[cur].pop();
      if (ret.first)
	return ret;
    }
    return galois::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(ApproxOrderByIntegerMetric)

template<class T, class Indexer = DummyIndexer<T>, typename ContainerTy = GFIFO<T>, bool concurrent=true >
class LogOrderByIntegerMetric : private boost::noncopyable {
  typename ContainerTy::template rethread<concurrent> data[sizeof(unsigned int)*8 + 1];

  Indexer I;
  substrate::PerThreadStorage<unsigned int> cursor;

  int num() {
    return sizeof(unsigned int)*8 + 1;
  }

  int getBin(unsigned int i) {
    if (i == 0) return 0;
    return sizeof(unsigned int)*8 - __builtin_clz(i);
  }

public:
  typedef T value_type;
  template<bool newconcurrent>
  struct rethread {
    typedef LogOrderByIntegerMetric<T, Indexer, ContainerTy, newconcurrent> type;
  };

  LogOrderByIntegerMetric(const Indexer& x = Indexer())
    :I(x)
  {
    for (int i = 0; i < cursor.size(); ++i)
      *cursor.getRemote(i) = 0;
  }

  void push(value_type val) {
    unsigned int index = I(val);
    index = getBin(index);
    return data[index].push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    unsigned int& cur = concurrent ? *cursor.getLocal() : *cursor.getRemote(0);
    galois::optional<value_type> ret = data[cur].pop();
    if (ret)
      return ret;

    //must move cursor
    for (cur = 0; cur < num(); ++cur) {
      ret = data[cur].pop();
      if (ret.first)
	return ret;
    }
    cur = 0;
    return galois::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(LogOrderByIntegerMetric)

template<typename T, typename Indexer = DummyIndexer<T>, typename LocalTy = GFIFO<T>, typename GlobalTy = GFIFO<T> >
class LocalFilter {
  GlobalTy globalQ;

  struct p {
    typename LocalTy::template rethread<false> Q;
    unsigned int current;
  };
  substrate::PerThreadStorage<p> localQs;
  Indexer I;

public:
  typedef T value_type;

  LocalFilter(const Indexer& x = Indexer()) : I(x) {
    for (int i = 0; i < localQs.size(); ++i)
      localQs.get(i).current = 0;
  }

  //! change the concurrency flag
  template<bool newconcurrent>
  struct rethread {
    typedef LocalFilter type;
  };

  //! push a value onto the queue
  void push(value_type val) {
    unsigned int index = I(val);
    p& me = localQs.get();
    if (index <= me.current)
      me.Q.push(val);
    else
      globalQ.push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      globalQ.push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    galois::optional<value_type> r = localQs.get().Q.pop();
    if (r)
      return r;

    r = globalQ.pop();
    if (r)
      localQs.get().current = I(r.second);
    return r;
  }
};
GALOIS_WLCOMPILECHECK(LocalFilter)

#if 0
//Bag per writer, reader steals entire bag
template<typename T, int chunksize = 64>
class MP_SC_Bag {
  class Chunk : public FixedSizeRing<T, chunksize, false>, public ConExtListNode<Chunk>::ListNode {};

  runtime::MM::FixedSizeAllocator heap;

  runtime::PerThreadStorage<PtrLock<Chunk*, true> > write_stack;

  ConExtLinkedStack<Chunk, true> read_stack;
  Chunk* current;

public:
  typedef T value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef MP_SC_Bag<T, chunksize> WL;
  };

  MP_SC_Bag() :heap(sizeof(Chunk)), current(0) {}

  bool push(value_type val) {
    PtrLock<Chunk*, true>& L = write_stack.get();
    L.lock();
    Chunk* OldL = L.getValue();
    if (OldL && OldL->push_back(val)) {
      L.unlock();
      return true;
    }
    Chunk* nc = new (heap.allocate(sizeof(Chunk))) Chunk();
    bool retval = nc->push_back(val);
    assert(retval);
    L.unlock_and_set(nc);
    if (OldL)
      read_stack.push(OldL);
    return true;
  }

  bool aborted(value_type val) {
    return push(val);
  }

  std::pair<bool, value_type> pop() {
    //Read stack
    if (!current)
      current = read_stack.pop();
    if (current)
      std::pair<bool, value_type> ret = current->pop_front();
    if (ret.first)
      return ret;
    //try taking from our write queue
    read_stack.steal(write_stack.get());
    ret = read_stack.pop();
    if (ret.first)
      return ret;
    //try stealing from everywhere
    for (int i = 0; i < write_stack.size(); ++i) {
      read_stack.steal(write_stack.get(i));
    }
    return read_stack.pop();
  }

  bool empty() {
    if (!read_stack.empty()) return false;
    for (int i = 0; i < write_stack.size(); ++i)
      if (!write_stack.get(i).empty())
	return false;
    return true;
  }


};
GALOIS_WLCOMPILECHECK(MP_SC_Bag)
#endif

//Per CPU and Per Level Queues, with staving flags
template<typename T, typename LocalWL, typename GlobalWL>
class RequestHierarchy {
public:
  typedef T value_type;

private:
  substrate::PerThreadStorage<typename LocalWL::template rethread<false> > localQueues;
  substrate::PerPackageStorage<GlobalWL> sharedQueues;
  substrate::PerPackageStorage<unsigned long> starvingFlags;
  GlobalWL gwl;
  unsigned long gStarving;

  //Clear the starving flag for all levels in which the thread
  //is a master thread of a level below
  void clearStarving() {
    gStarving = 0;
    *starvingFlags.getLocal() = 0;
  }

public:
  void push(value_type val) {
    if (gStarving)
      gwl.push(val);
    else if (*starvingFlags.getLocal())
      sharedQueues.getLocal()->push(val);
    else
      localQueues.getLocal()->push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    //Try the local queue first
    galois::optional<value_type> ret = localQueues.getLocal()->pop();
    if (ret)
      return ret;

    //check parent
    ret = sharedQueues.getLocal()->pop();
    if (ret) {
      clearStarving();
      return ret;
    }

    //check global
    ret = gwl.pop();
    if (ret) {
      clearStarving();
      return ret;
    }

    //Any thread can set the package starving flag
    *starvingFlags.getLocal() = 1;
    //if we are master for the package, handle flags
    if (sharedQueues.isFirstInLevel())
      ; // TODO: ???
    return ret;
  }
};
GALOIS_WLCOMPILECHECK(RequestHierarchy)

template<typename T, typename LocalWL, typename DistPolicy>
class ReductionWL {
  typedef substrate::CacheLineStorage<LocalWL> paddedLocalWL;

  paddedLocalWL* WL;
  GFIFO<T> backup;
  int starving;

public:
  typedef T value_type;

  ReductionWL() :starving(0) {
    WL = new paddedLocalWL[DistPolicy::getNumIslands()];
  }

  ~ReductionWL() {
    delete[] WL;
    WL = 0;
  }

  void push(value_type val) {
    if (starving)
      backup.push(val);
    else
      WL[DistPolicy::getThreadIsland()].data.push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    int myIsland = DistPolicy::getThreadIsland();
    galois::optional<value_type> val = WL[myIsland].data.pop();
    if (val || !DistPolicy::isThreadMaster())
      return val;

    int IFlag = 1 << myIsland;

    val = backup.pop();
    if (val.first) {
      __sync_fetch_and_and(&starving, ~IFlag);
      return val;
    }
    if (!starving & IFlag)
      __sync_fetch_and_or(&starving, IFlag);
    return val;
  }
};
GALOIS_WLCOMPILECHECK(ReductionWL)


#if 0
template<class T, class Indexer, typename ContainerTy = FIFO<T>, bool concurrent=true>
class dOrderByIntegerMetric : private boost::noncopyable {

  struct LevelItem {
    unsigned int cur_pri;
    unsigned int set_pri;
    ContainerTy* cur;
  };

  PerLevel<LevelItem> items;
  unsigned int global_min;

  std::map<unsigned int, ContainerTy*> data;
  SimpleLock<int, concurrent> mapLock;

public:

  typedef T value_type;
  template<bool newconcurrent>
  struct rethread {
    typedef dOrderByIntegerMetric<T, Indexer, typename ContainerTy::template rethread<newconcurrent>, newconcurrent> WL;
  };

  dOrderByIntegerMetric(const Indexer& x = Indexer())
    :I(x), numActive(ThreadPool::getActiveThreads())
  {

  }

  bool push(value_type val) {
    unsigned int index = I(val);
    data[index].push(val);
  }

  std::pair<bool, value_type> pop() {
    // print();
    unsigned int& cur = cursor.get();
    std::pair<bool, value_type> ret = data[cur].pop();
    if (ret.first)
      return ret;

    //must move cursor
    for (int i = 0; i < (num() + active() - 1) / active(); ++i) {
      cur += active();
      if (cur >= num())
	cur %= active();
      ret = data[cur].try_pop();
      if (ret.first)
	return ret;
    }
    return std::pair<bool, value_type>(false, value_type());
  }

  std::pair<bool, value_type> try_pop() {
    return pop();
  }

  bool empty() {
    for (unsigned int i = 0; i < num(); ++i)
      if (!data[i].empty())
	return false;
    return true;
  }

  bool aborted(value_type val) {
    return push(val);
  }

};
#endif

#if 0
template<typename T>
class GWL_ChaseLev_Dyn : private boost::noncopyable {

  struct DequeNode {
    enum { ArraySize = 256 };
    T itsDataArr[ArraySize];
    DequeNode* next;
    DequeNode* prev;
  };

  // struct BottomStruct {
  //   DequeNode* nodeP;
  //   int32_t cellIndex;
  // };

  // struct TopStruct {
  //   DequeNode* nodeP;
  //   int32_t cellIndex;
  //   int32_t tag;
  // };

  //FIXME: generalize for 32 bit hosts too
  volatile uint64_t Bottom;
  volatile uint64_t Top;

  uint64_t ReadBottom() {
    //This read may need to be atomic
    return Bottom;
  }

  uint64_t ReadTop() {
    //This read may need to be atomic
    return Top;
  }

  void WriteBottom(uint64_t V) {
    //This write may need to be atomic
    Bottom = V;
  }
  void WriteTop(uint64_t V) {
    //This write may need to be atomic
    Top = V;
  }

  //Take advantage of the 48 bit addresses on amd64

  void DecodeBottom(uint64_t v, DequeNode*& currNode, uint8_t& currIndex) {
    currNode  =      (DequeNode*)(v & 0x0000FFFFFFFFFFFFULL);
    currIndex = (uint8_t)((v >> 48) & 0x0FFFF);
  }
  uint64_t EncodeBottom(DequeNode* currNode, uint8_t currIndex) {
    uint64_t v = 0;
    v  = (uint64_t)currNode;
    v |= (uint64_t)currIndex << 48;
    return v;
  }

  void DecodeTop(uint64_t v, uint8_t& currTopTag, DequeNode*& currTopNode, uint8_t& currTopIndex) {
    currTopNode  =      (DequeNode*)(v & 0x0000FFFFFFFFFFFFULL);
    currTopIndex = (uint8_t)((v >> 48) & 0x0FFFF);
    currTopTag   = (uint8_t)((v >> 56) & 0x0FFFF);
  }
  uint64_t EncodeTop(uint8_t currTopTag, DequeNode* currTopNode, uint8_t currTopIndex) {
    uint64_t v = 0;
    v  = (uint64_t)currTopNode;
    v |= (uint64_t)currTopIndex << 48;
    v |= (uint64_t)currTopTag << 56;
    return v;
  }

  bool CAS(volatile uint64_t* ptr, uint64_t old, uint64_t val) {
    return __sync_bool_compare_and_swap(ptr,old,val);
  }

  DequeNode* AllocateNode() {
    return new DequeNode;
  }

  bool emptinessTest(uint64_t bottomVal, uint64_t topVal) {
    DequeNode* botNode = 0;
    uint8_t botCellIndex = 0;
    DecodeBottom(bottomVal,botNode,botCellIndex);
    uint8_t topTag = 0;
    DequeNode* topNode = 0;
    uint8_t topCellIndex = 0;
    DecodeTop(topVal, topTag,topNode,topCellIndex);
    if ((botNode==topNode) && (botCellIndex==topCellIndex ||
			       botCellIndex==(topCellIndex+1))) {
      return true;
    } else if ((botNode==topNode->next) && (botCellIndex==0) &&
	       (topCellIndex==(DequeNode::ArraySize-1))) {
      return true;
    }
    return false;
  }

  //Local
  void PushBottom(T theData) {
    DequeNode* currNode = 0;
    uint8_t currIndex = 0;
    DequeNode* newNode = 0;
    uint8_t newIndex = 0;
    DecodeBottom(ReadBottom(),currNode, currIndex);
    currNode->itsDataArr[currIndex] = theData;
    //      std::cerr << "[" << currIndex << "] " << newIndex << "\n";
    //      std::cerr << "Wrote: " << theData << "\n";
    if (currIndex != 0) {
      newNode = currNode;
      newIndex = currIndex - 1;
    } else {
      newNode = AllocateNode();
      newNode->next = currNode;
      currNode->prev = newNode;
      newIndex = DequeNode::ArraySize - 1;
    }
    //	std::cerr << currIndex << " " << newIndex << "\n" << std::endl;
    WriteBottom(EncodeBottom(newNode,newIndex));
  }

  //Remote
  T PopTop(bool& EMPTY, bool& ABORT) {
    EMPTY = false;
    ABORT = false;
    uint64_t currTop = ReadTop();
    uint8_t currTopTag = 0;
    uint8_t currTopIndex = 0;
    DequeNode* currTopNode = 0;
    uint8_t newTopTag = 0;
    uint8_t newTopIndex = 0;
    DequeNode* newTopNode = 0;
    DequeNode* nodeToFree = 0;
    DecodeTop(currTop, currTopTag, currTopNode, currTopIndex);
    uint64_t currBottom = ReadBottom();
    if (emptinessTest(currBottom, currTop)) {
      if (currTop == ReadTop()) {
	EMPTY = true;
	return T();
      } else {
	ABORT = true;
	return T();
      }
    }
    if (currTopIndex != 0) {
      newTopTag = currTopTag;
      newTopNode = currTopNode;
      newTopIndex = currTopIndex - 1;
    } else {
      nodeToFree = currTopNode->next;
      newTopTag = currTopTag + 1;
      newTopNode = currTopNode->prev;
      newTopIndex = DequeNode::ArraySize - 1;
    }
    uint64_t newTopVal = EncodeTop(newTopTag, newTopNode, newTopIndex);
    T retVal = currTopNode->itsDataArr[currTopIndex];
    if (CAS(&Top, currTop, newTopVal)) {
      if (nodeToFree)
	delete nodeToFree;
      return retVal;
    } else {
      ABORT = true;
      return T();
    }
  }

  //Local only
  bool Empty() {
    return emptinessTest(ReadBottom(), ReadTop());
  }

  //Local
  T PopBottom(bool& EMPTY) {
    EMPTY = false;
    DequeNode* oldBotNode = 0;
    uint8_t oldBotIndex = 0;
    DequeNode* newBotNode = 0;
    uint8_t newBotIndex = 0;
    uint64_t oldBotVal = ReadBottom();
    DecodeBottom(oldBotVal, oldBotNode, oldBotIndex);
    if (oldBotIndex != DequeNode::ArraySize-1) {
      newBotNode = oldBotNode;
      newBotIndex = oldBotIndex+1;
    } else {
      newBotNode = oldBotNode->next;
      newBotIndex = 0;
    }
    //      std::cerr << oldBotIndex << " [" << newBotIndex << "]\n";
    uint64_t newBotVal = EncodeBottom(newBotNode,newBotIndex);
    WriteBottom(newBotVal);
    uint64_t currTop = ReadTop();
    uint8_t currTopTag = 0;
    DequeNode* currTopNode = 0;
    uint8_t currTopIndex = 0;
    DecodeTop(currTop, currTopTag,currTopNode,currTopIndex);
    T retVal = newBotNode->itsDataArr[newBotIndex]; // Read data to be popped
    //      std::cerr << "Read: " << retVal << "\n";
    if (oldBotNode == currTopNode && oldBotIndex == currTopIndex ) {
      // Case 1: if Top has crossed Bottom
      //Return bottom to its old possition:
      WriteBottom(EncodeBottom(oldBotNode,oldBotIndex));
      EMPTY = true;
      //	std::cerr << "Returning empty 1\n";
      return T();
    } else if ( newBotNode == currTopNode && newBotIndex == currTopIndex ) {
      // Case 2: When popping the last entry
      //in the deque (i.e. deque is
      //empty after the update of bottom).
      //Try to update Top’s tag so no concurrent PopTop operation will also pop the same entry:
      uint64_t newTopVal = EncodeTop(currTopTag+1, currTopNode, currTopIndex);
      if (CAS(&Top, currTop, newTopVal)) {
	if (oldBotNode != newBotNode)
	  delete oldBotNode;
	return retVal;
      } else {
	// if CAS failed (i.e. a concurrent PopTop operation already popped the last entry):
	//Return bottom to its old possition:
	WriteBottom(EncodeBottom(oldBotNode,oldBotIndex));
	EMPTY = true;
	//	  std::cerr << "Returning empty 2\n";
	return T();
      }
    } else {
      // Case 3: Regular case (i.e. there was at least one entry in the deque after bottom’s update):
      if (oldBotNode != newBotNode)
	delete oldBotNode;
      return retVal;
    }
  }

public:

  typedef GWL_ChaseLev_Dyn<T> ConcurrentTy;
  typedef GWL_ChaseLev_Dyn<T> SingleTy;
  enum {MAYSTEAL = true};


  GWL_ChaseLev_Dyn()
    :Bottom(0), Top(0)
  {
    DequeNode* nodeA = AllocateNode();
    DequeNode* nodeB = AllocateNode();
    nodeA->prev = 0;
    nodeA->next = nodeB;
    nodeB->next = 0;
    nodeB->prev = nodeA;
    int newIndex = DequeNode::ArraySize - 1;
    WriteBottom(EncodeBottom(nodeA,newIndex));
    WriteTop(EncodeTop(0,nodeA,newIndex));
  }


  //These should only be called by one thread
  void push(T val) {
    PushBottom(val);
  }

  T pop(bool& succeeded) {
    bool Emp;
    T retval = PopBottom(Emp);
    succeeded = !Emp;
    return retval;
  }

  //This can be called by any thread
  T steal(bool& succeeded) {
    bool Empty, Abort;
    T retval = PopTop(Empty,Abort);
    succeeded = !(Empty || Abort);
    return retval;
  }

  bool empty() {
    return Empty();
  }

};

template<typename T>
class GWL_Idempotent_LIFO : private boost::noncopyable {

  packedInt2<32,32> anchor; //tail,tag
  unsigned int capacity;
  T* volatile tasks;

  inline void order() {
    //Compiler barier
    __asm__("":::"memory");
  }

  bool Empty() {
    unsigned int t,g;
    anchor.packedRead(t,g);
    return t == 0;
  }

  void put(T t_ask) {
    //Order write in 3 before write in 4
    unsigned int t,g;
    anchor.packedRead(t,g);
    if (t == capacity) {
      expand();
      put(t_ask);
      return;
    }
    tasks[t] = t_ask;
    order();
    anchor.packedWrite(t+1,g+1);
  }

  T take(bool& EMPTY) {
    EMPTY = false;
    unsigned int t,g;
    anchor.packedRead(t,g);
    if (t == 0) {
      EMPTY = true;
      return T();
    }
    T t_ask = tasks[t-1];
    anchor.packedWrite(t-1,g);
    return t_ask;
  }

  T i_steal(bool& EMPTY) {
    EMPTY = false;
    //Order read in 1 before read in 3
    //Order read in 4 before CAS in 5
    unsigned int t,g;
    anchor.packedRead(t,g);
    if (t == 0) {
      EMPTY = true;
      return T();
    }
    order();
    T* a = tasks;
    T t_ask = a[t-1];
    order();
    if (!anchor.CAS(t,g, t-1,g )) {
      return i_steal(EMPTY);
    }
    return t_ask;
  }

  void expand() {
    //Order writes in 2 before write in 3
    //Order write in 3 before write in put:4
    T* a = new T[2*capacity];
    for( int i = 0; i < (int)capacity; ++i)
      a[i] = tasks[i];
    order();
    tasks = a;
    capacity = 2*capacity;
    order();
  }

public:
  typedef GWL_Idempotent_LIFO<T> ConcurrentTy;
  typedef GWL_Idempotent_LIFO<T> SingleTy;
  enum {MAYSTEAL = true};

  GWL_Idempotent_LIFO(int size = 256)
    :anchor(0,0), capacity(size)
  {
    tasks = new T[size];
  }

  void push(T val) {
    put(val);
  }

  T pop(bool& succeeded) {
    bool Empty;
    T retval = take(Empty);
    succeeded = !Empty;
    return retval;
  }

  //This can be called by any thread
  T steal(bool& succeeded) {
    bool Empty;
    T retval = i_steal(Empty);
    succeeded = !Empty;
    return retval;
  }

  bool empty() {
    return Empty();
  }

};

template<typename T>
class GWL_Idempotent_FIFO_SB : private boost::noncopyable {

  struct TaskArrayWithSize {
    int size;
    T array[1];
  };

  TaskArrayWithSize* mkArray(int num) {
    TaskArrayWithSize* r = (TaskArrayWithSize*)malloc(sizeof(TaskArrayWithSize)+sizeof(T[num]));
    r->size = num;
    return r;
  }

  packedInt3<21,21,22> anchor;
  TaskArrayWithSize* volatile tasks;

  inline void order() {
    //Compiler barier
    __asm__("":::"memory");
  }


  bool Empty() {
    unsigned int h,s,g;
    anchor.packedRead(h,s,g);
    return s == 0;
  }

  void put(T t_ask) {
    //Order write in 3 before write in 4
    unsigned int h,s,g;
    anchor.packedRead(h,s,g);
    if ((int)s == tasks->size) {
      expand();
      put(t_ask);
      return;
    }
    tasks->array[(h+s)%tasks->size] = t_ask;
    order();
    anchor.packedWrite(h,s+1,g+1);
  }

  T take(bool& EMPTY) {
    EMPTY = false;
    unsigned int h,s,g;
    anchor.packedRead(h,s,g);
    if (s == 0) {
      EMPTY = true;
      return T();
    }
    T t_ask = tasks->array[(h+s-1)%tasks->size];
    anchor.packedWrite(h,s-1,g);
    return t_ask;
  }

  T i_steal(bool& EMPTY) {
    EMPTY = false;
    //Order read in 1 before read in 3
    //Order read in 4 before CAS in 6
    unsigned int h,s,g;
    anchor.packedRead(h,s,g);
    if (s == 0) {
      EMPTY = 0;
      return T();
    }
    order();
    TaskArrayWithSize* a = tasks;
    T t_ask = a->array[h%a->size];
    unsigned int h2 = (h+1) % a->size;
    order();
    if (!anchor.CAS(h,s,g , h2,s-1,g )) {
      return i_steal(EMPTY);
    }
    return t_ask;
  }

  void expand() {
    //Order writes in 2 and 4 before write in 5
    //Order write in 5 before write in put:4
    unsigned int h,s,g;
    anchor.packedRead(h,s,g);
    TaskArrayWithSize* a = mkArray(2*s);
    for (unsigned int i = 0; i < s; ++i)
      a->array[(h+i)%a->size] = tasks->array[(h+i)%tasks->size];
    order();
    tasks = a;
    order();
  }

public:
  typedef GWL_Idempotent_FIFO_SB<T> ConcurrentTy;
  typedef GWL_Idempotent_FIFO_SB<T> SingleTy;
  enum {MAYSTEAL = true};

  GWL_Idempotent_FIFO_SB()
    :anchor(0,0,0)
  {
    //MAGIC INITIAL SIZE
    tasks = mkArray(256);
  }

  void push(T val) {
    put(val);
  }

  T pop(bool& succeeded) {
    bool Empty;
    T retval = take(Empty);
    succeeded = !Empty;
    return retval;
  }

  //This can be called by any thread
  T steal(bool& succeeded) {
    bool Empty;
    T retval = i_steal(Empty);
    succeeded = !Empty;
    return retval;
  }

  bool empty() {
    return Empty();
  }

};

#endif


template<typename T>
class GWL_Idempotent_FIFO: private boost::noncopyable {

  struct TaskArrayWithSize {
    int size;
    T array[1];
  };

  TaskArrayWithSize* mkArray(int num) {
    TaskArrayWithSize* r = (TaskArrayWithSize*)malloc(sizeof(TaskArrayWithSize)+sizeof(T[num]));
    r->size = num;
    return r;
  }

  int head;
  int tail;
  TaskArrayWithSize* volatile tasks;

  inline void order() {
    //Compiler barier
    __asm__("":::"memory");
  }

  bool Empty() {
    return head == tail;
  }

  void put(T t_ask) {
    //Order write at 4 before write at 5
    int h = head;
    int t = tail;
    if (t == h+tasks->size) {
      expand();
      put(t_ask);
      return;
    }
    tasks->array[t%tasks->size] = t_ask;
    order();
    tail = t+1;
  }

  T take(bool& EMPTY) {
    EMPTY = false;
    int h = head;
    int t = tail;
    if (h == t) {
      EMPTY = true;
      return T();
    }
    T t_ask = tasks->array[h%tasks->size];
    head = h+1;
    return t_ask;
  }

  T i_steal(bool& EMPTY) {
    EMPTY = false;
    //Order read in 1 before read in 2
    //Order read in 1 before read in 4
    //Order read in 5 before CAS in 6
    int h = head;
    order();
    int t = tail;
    order();
    if (h == t) {
      EMPTY = true;
      return T();
    }
    TaskArrayWithSize* a = tasks;
    T t_ask = a->array[h%a->size];
    order();
    if (!__sync_bool_compare_and_swap(&head,h,h+1)) {
      return i_steal(EMPTY);
    }
    return t_ask;
  }

  void expand() {
    //Order writes in 2 and 4 before write in 5
    //Order write in 5 before write in put:5
    int size = tasks->size;
    TaskArrayWithSize* a = mkArray(2*size);
    for (int i = head; i < tail; ++i)
      a->array[i%a->size] = tasks->array[i%tasks->size];
    order();
    tasks = a;
    order();
  }

public:
  typedef T value_type;
  typedef GWL_Idempotent_FIFO<T> ConcurrentTy;
  typedef GWL_Idempotent_FIFO<T> SingleTy;
  enum {MAYSTEAL = true};

  GWL_Idempotent_FIFO(int size = 256): head(0), tail(0) {
    tasks = mkArray(size);
  }

  void push(value_type val) {
    put(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    bool Empty;
    T retval = take(Empty);
    return !Empty ?
      galois::optional<value_type>(retval) :
      galois::optional<value_type>();
  }

  //This can be called by any thread
  T steal(bool& succeeded) {
    bool Empty;
    T retval = i_steal(Empty);
    succeeded = !Empty;
    return retval;
  }
};
GALOIS_WLCOMPILECHECK(GWL_Idempotent_FIFO)

template<typename Partitioner = DummyIndexer<int>, typename T = int, typename ChildWLTy = dChunkedFIFO<>, bool concurrent=true>
class PartitionedWL : private boost::noncopyable {

  Partitioner P;
  substrate::PerThreadStorage<ChildWLTy> Items;
  int active;

public:
  template<bool newconcurrent>
  using rethread = PartitionedWL<T, Partitioner, ChildWLTy, newconcurrent>;

  typedef T value_type;

  PartitionedWL(const Partitioner& p = Partitioner()) :P(p), active(galois::getActiveThreads()) {
    //std::cerr << active << "\n";
  }

  void push(value_type val)  {
    unsigned int index = P(val);
    //std::cerr << "[" << index << "," << index % active << "]\n";
    return Items.get(index % active).push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    galois::optional<value_type> r = Items.get().pop();
    // std::cerr << "{" << Items.myEffectiveID() << "}";
    // if (r.first)
    //   std::cerr << r.first;
    return r;
  }

  galois::optional<value_type> try_pop() {
    return pop();
  }
};
GALOIS_WLCOMPILECHECK(PartitionedWL)

template<class Compare = std::less<int>, typename T = int>
class SkipListQueue : private boost::noncopyable {

  galois::ConcurrentSkipListMap<T,int,Compare> wl;
  int magic;

public:
  template<bool newconcurrent>
  using rethread = SkipListQueue<Compare, T>;

  template<typename Tnew>
  using retype = SkipListQueue<Compare, Tnew>;

  typedef T value_type;

  bool push(value_type val) {
    wl.putIfAbsent(val, &magic);
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
    return true;
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    return wl.pollFirstKey();
  }
};
GALOIS_WLCOMPILECHECK(SkipListQueue)


template<class Compare = std::less<int>, typename T = int, bool concurrent = true>
class SetQueue : private boost::noncopyable, private substrate::PaddedLock<concurrent> {
  std::set<T, Compare, runtime::FixedSizeAllocator<T> > wl;

  using substrate::PaddedLock<concurrent>::lock;
  using substrate::PaddedLock<concurrent>::try_lock;
  using substrate::PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  using rethread = SetQueue<Compare, T, newconcurrent>;

  template<typename Tnew>
  using retype = SetQueue<Compare, Tnew, concurrent>;

  typedef T value_type;

  void push(value_type val) {
    lock();
    wl.insert(val);
    unlock();
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    lock();
    while (b != e)
      wl.insert(*b++);
    unlock();
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    if (substrate::ThreadPool::getTID() == 0)
      push(range.begin(), range.end());
  }

  galois::optional<value_type> pop() {
    lock();
    if (wl.empty()) {
      unlock();
      return galois::optional<value_type>();
    }
    value_type r = *wl.begin();
    wl.erase(wl.begin());
    unlock();
    return galois::optional<value_type>(r);
  }
};
GALOIS_WLCOMPILECHECK(SetQueue)

template<class Compare = std::less<int>, typename T = int>
class FCPairingHeapQueue : private boost::noncopyable {
  galois::FCPairingHeap<T,Compare> wl;

public:
  template<bool newconcurrent>
  using rethread = FCPairingHeapQueue<Compare, T>;

  template<typename Tnew>
  using retype = FCPairingHeapQueue<Compare, Tnew>;

  typedef T value_type;

  bool push(value_type val) {
    wl.add(val);
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
    return true;
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    return wl.pollMin();
  }
};
GALOIS_WLCOMPILECHECK(FCPairingHeapQueue)


template<class Indexer, typename ContainerTy = galois::worklists::GFIFO<>, typename T = int, bool concurrent = true >
class SimpleOrderedByIntegerMetric : private boost::noncopyable, private galois::substrate::PaddedLock<concurrent> {

   using galois::substrate::PaddedLock<concurrent>::lock;
   using galois::substrate::PaddedLock<concurrent>::try_lock;
   using galois::substrate::PaddedLock<concurrent>::unlock;

  typedef ContainerTy CTy;

  CTy* current;
  int cint;
  std::map<int, CTy*> mapping;
  Indexer I;

  CTy* updateLocalOrCreate(int i) {
    //Try local then try update then find again or else create and update the master log
    //check if current bin is the right thing to use (lock-free)
    if (i == cint) {
      CTy* n = current;
      if (n) return n;
    }
    lock();
    CTy*& lC = mapping[i];
    if (!lC)
      lC = new CTy();
    unlock();
    return lC;
  }

 public:
  template<bool newconcurrent>
  using rethread = SimpleOrderedByIntegerMetric<Indexer,ContainerTy,T,newconcurrent>;

  template<typename Tnew>
  using retype = SimpleOrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>,Tnew,concurrent>;

  typedef T value_type;

  SimpleOrderedByIntegerMetric(const Indexer& x = Indexer())
    :current(0), I(x)
  { }

  void push(value_type val) {
    unsigned int index = I(val);
    CTy* lC = updateLocalOrCreate(index);
    lC->push(val);
  }

  template<typename ItTy>
  bool push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
    return true;
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    //Fastpath
    CTy* c = current;
    galois::optional<value_type> retval;
    if (c && (retval = c->pop()))
      return retval;

    //Failed, find minimum bin
    lock();
    if (current != c) {
      unlock();
      return pop();
    }
    for (typename std::map<int, CTy*>::iterator ii = mapping.begin(), ee = mapping.end();
        ii != ee; ++ii) {
      current = ii->second;
      if ((retval = current->pop())) {
        cint = ii->first;
        goto exit;
      }
    }
    exit:
      unlock();
    return retval;
  }
};

template<class Indexer, typename ContainerTy = galois::worklists::ChunkedLIFO<16>, bool BSP = true, typename T = int, bool concurrent = true>
class CTOrderedByIntegerMetric : private boost::noncopyable {
  typedef typename ContainerTy::template rethread<concurrent> CTy;

  struct perItem {
    CTy* current;
    unsigned int curVersion;
    unsigned int scanStart;
    perItem() :current(NULL), curVersion(0), scanStart(0) {}
  };

  // struct mhc {
  //   static size_t hash(int x) { return x; }
  //   static bool equal(const int& x, const int& y) { return x == y; }
  // };
  //typedef tbb::concurrent_hash_map<int, CTy*, mhc> HM;
  //HM wl;
  // typedef ADLSkipList<CTy> SL;
  // SL wl;
  galois::ConcurrentSkipListMap<int,CTy> wl;
  int maxV;

  Indexer I;
  substrate::PerThreadStorage<perItem> current;

  CTy* getOrCreate(int index) {
    CTy* r = wl.get(index);
    if (r) return r;
    CTy* r2 = new CTy;
    r = wl.putIfAbsent(index, r2);
    if (r)
      delete r2;
    else
      r = r2;
    return r;
  }

 public:
  template<bool newconcurrent>
  using rethread = CTOrderedByIntegerMetric<Indexer,ContainerTy,BSP,T,newconcurrent>;

  template<typename Tnew>
  using retype = CTOrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>,BSP,Tnew,concurrent>;

  typedef T value_type;

 CTOrderedByIntegerMetric(const Indexer& x = Indexer())
   :maxV(1),I(x)
  {}

  void push(value_type val) {
    unsigned int index = I(val);
    perItem& pI = *current.getLocal();
    //fastpath
    if (index == pI.curVersion && pI.current) {
      pI.current->push(val);
      return;
    }
    //slow path
    CTy* lC = getOrCreate(index);
    if (BSP && index < pI.scanStart)
      pI.scanStart = index;
    //opportunistically move to higher priority work
    if (index < pI.curVersion) {
      pI.curVersion = index;
      pI.current = lC;
    }
    lC->push(val);
    //update max
    unsigned int oldMax;
    while ((oldMax = maxV) < index)
      __sync_bool_compare_and_swap(&maxV, oldMax, index);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    //Find a successful pop
    perItem& pI = *current.getLocal();
    CTy*& C = pI.current;
    galois::optional<value_type> retval;
    if (C && (retval = C->pop()))
      return retval;

    //Failed, find minimum bin
    bool localLeader = substrate::ThreadPool::isLeader();

    unsigned msS = 0;
    if (BSP) {
      msS = pI.scanStart;
      if (localLeader)
	for (int i = 0; i < (int) galois::getActiveThreads(); ++i)
	  msS = std::min(msS, current.getRemote(i)->scanStart);
      else
	msS = std::min(msS, current.getRemote(substrate::ThreadPool::getLeader())->scanStart);
    }

    for (int i = msS; i <= maxV; ++i) {
      // typename HM::const_accessor a;
      // if (wl.find(a, i)) {
      CTy* w = wl.get(i);
      if (w) {
	pI.curVersion = i;
	pI.scanStart = i;
	C = w; //a->second;
	if (C && (retval = C->pop()))
	  return retval;
      }
    }
    return galois::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(CTOrderedByIntegerMetric)

#if _POSIX_BARRIERS > 0
template<class Indexer, typename ContainerTy, bool concurrent = true, int binmax = 1024*1024 >
class BarrierOBIM : private boost::noncopyable {
  typedef typename ContainerTy::template rethread<concurrent> CTy;

  volatile unsigned int current;
  volatile unsigned int pushmax;

  CTy* B;

  Indexer I;

  substrate::TerminationDetection& term;
  pthread_barrier_t barr1;
  pthread_barrier_t barr2;

 public:
  template<bool newconcurrent>
  using rethread = BarrierOBIM<Indexer,ContainerTy,newconcurrent,binmax>;

  template<typename Tnew>
  using retype = BarrierOBIM<Indexer,typename ContainerTy::template retype<Tnew>,concurrent, binmax>;

  typedef typename CTy::value_type value_type;

  BarrierOBIM(const Indexer& x = Indexer())
    :current(0), pushmax(0), I(x), term(substrate::getSystemTermination(galois::getActiveThreads()))
  {
    B = new CTy[binmax];
    //std::cerr << "$"<<getThreadPool().getActiveThreads() <<"$";
    pthread_barrier_init(&barr1, NULL, galois::getActiveThreads());
    pthread_barrier_init(&barr2, NULL, galois::getActiveThreads());
  }
  ~BarrierOBIM() {
    delete[] B;
  }

  void push(value_type val) {
    unsigned int index = I(val);
    //std::cerr << "P: " << index << "\n";
    if (index < current)
      index = current;
    if (index >= binmax)
      index = binmax - 1;
    term.localTermination(true);
    unsigned int oldi;
    while (index > (oldi = pushmax)) {
      __sync_bool_compare_and_swap(&pushmax, oldi, index);
    }
    B[index].push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    pthread_barrier_wait(&barr1);
    term.initializeThread();
    pthread_barrier_wait(&barr2);
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    do {
      if (current > pushmax)
	return galois::optional<value_type>();
      do {
	//Find a successful pop
	galois::optional<value_type> retval = B[current].pop();
	if (retval) {
	  term.localTermination(true);
	  return retval;
	}
	term.localTermination(false);
      } while (!term.globalTermination());

      pthread_barrier_wait(&barr1);
      term.initializeThread();
      if (galois::substrate::ThreadPool::getTID() == 0) {
	//std::cerr << "inc: " << current << "\n";
	if (current <= pushmax)
	  __sync_fetch_and_add(&current, 1);
      }

      pthread_barrier_wait(&barr2);
    } while (true);
  }
};
#endif

template<typename T = int, bool concurrent = true>
class Random : private boost::noncopyable {
  typedef typename std::conditional<concurrent, substrate::SimpleLock, substrate::DummyLock>::type LockTy;
  std::pair<LockTy, std::deque<T> > wl[128];
  struct rstate { unsigned short d[3]; };
  substrate::PerThreadStorage<rstate > seeds;

  unsigned int nextRand() {
    return nrand48(seeds.getLocal()->d);
  }

  unsigned int pickBucket() {
    return nrand48(seeds.getLocal()->d) % 128;
  }

public:
  Random() { }

  template<bool newconcurrent>
  using rethread = Random<T, newconcurrent>;

  template<typename Tnew>
  using retype = Random<Tnew, concurrent>;

  typedef T value_type;

  void push(value_type val) {
    auto& entry = wl[pickBucket()];
    entry.first.lock();
    entry.second.push_back(val);
    entry.first.unlock();
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    unsigned b = pickBucket();
    for (int i = 0; i < 128; ++i) {
      auto& entry = wl[(i + b) % 128];
      entry.first.lock();
      if (entry.second.empty()) {
        entry.first.unlock();
      } else {
        size_t size = entry.second.size();
        unsigned int index = nextRand() % size;
        value_type retval = entry.second[index];
        std::swap(entry.second[index], entry.second[size-1]);
        entry.second.pop_back();
        entry.first.unlock();
        return galois::optional<value_type>(retval);
      }
    }
    return galois::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(Random)


template <typename T> struct GETID {
  int operator()(const T& v) {
    return v.getID();
  }
};
// This is a pointer to T specialization!
template <typename T> struct GETID<T*> {
  unsigned operator()(const T* v) {
    return (unsigned)(((uintptr_t) v) >> 7);
  }
};

#ifdef USE_TBB
template<typename T = int>
class TbbFIFO : private boost::noncopyable  {
  tbb::concurrent_bounded_queue<T> wl;

public:
  template<bool newconcurrent>
  using rethread = TbbFIFO<T>;

  template<typename Tnew>
  using retype = TbbFIFO<Tnew>;

  typedef T value_type;

  void push(value_type val) {
    wl.push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      wl.push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    wl.push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    T V = T();
    return wl.try_pop(V) ?
      galois::optional<value_type>(V) :
      galois::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(TbbFIFO)

//partitioned tbb
template<class Compare = std::less<int>, typename T = int>
class PTbb : private boost::noncopyable {
  typedef tbb::concurrent_priority_queue<T,Compare> TBBTy;

  struct PTD {
    TBBTy wl;
    LIFO<T> inq;
    drand48_data r;
  };

  galois::runtime::PerThreadStorage<PTD> tld;

  void pull_in(PTD* N) {
    galois::optional<T> r;
    while ((r = N->inq.pop()))
      N->wl.push(*r);
  }

public:

  PTbb() {
    for (size_t i = 0; i < tld.size(); ++i)
      srand48_r(i, &tld.getRemote(i)->r);
  }

  template<bool newconcurrent>
  using rethread = PTbb<Compare, T>;

  template<typename Tnew>
  using retype = PTbb<Compare, Tnew>;

  typedef T value_type;

  void push(value_type val) {
    //unsigned index = GETID<value_type>()(val);
    //index = (index & 0x00FF) ^ ((index >> 8) & 0x00FF);
    //index %= galois::getActiveThreads();
    // PTD* N = tld.getLocal();
    // if (index == runtime::LL::getTID()) {
    //   N->wl.push(val);
    // } else {
    //   tld.getRemote(index)->wl.push(val);
    // }
    long int index;
    lrand48_r(&tld.getLocal()->r, &index);
    index %= galois::getActiveThreads();
    if (index == runtime::LL::getTID())
      tld.getLocal()->wl.push(val);
    else
      tld.getRemote(index)->inq.push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    PTD* N = tld.getLocal();
    long int index;
    unsigned loc = runtime::LL::getTID();
    while (b != e) {
      lrand48_r(&N->r, &index);
      index %= galois::getActiveThreads();
      if (index == loc)
	N->wl.push(*b++);
      else
	tld.getRemote(index)->inq.push(*b++);
    }
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    value_type retval;
    PTD* N = tld.getLocal();
    pull_in(N);
    if (N->wl.try_pop(retval)) {
      return galois::optional<value_type>(retval);
    } else {
      return galois::optional<value_type>();
    }
  }
};

//stealing tbb
template<class Compare = std::less<int>, typename T = int>
class STbb : private boost::noncopyable {
  typedef tbb::concurrent_priority_queue<T,Compare> TBBTy;

  galois::runtime::PerThreadStorage<TBBTy> tld;

public:
  template<bool newconcurrent>
  using rethread = STbb<Compare, T>;

  template<typename Tnew>
  using retype = STbb<Compare, Tnew>;

  typedef T value_type;

  void push(value_type val) {
    tld.getLocal()->push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    value_type retval;
    if (tld.getLocal()->try_pop(retval))
      return retval;
    for (unsigned i = 0; i < galois::getActiveThreads(); ++i) {
      if (tld.getRemote(i)->try_pop(retval))
	return retval;
    }
    return galois::optional<value_type>();
  }
};

template<class Compare = std::less<int>, typename T = int>
class TbbPriQueue : private boost::noncopyable {
  tbb::concurrent_priority_queue<T,Compare> wl;

public:
  template<bool newconcurrent>
  using rethread = TbbPriQueue<Compare, T>;

  template<typename Tnew>
  using retype = TbbPriQueue<Compare, Tnew>;

  typedef T value_type;

  void push(value_type val) {
    wl.push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    push(range.local_begin(), range.local_end());
  }

  galois::optional<value_type> pop() {
    value_type retval;
    if (wl.try_pop(retval)) {
      return galois::optional<value_type>(retval);
    } else {
      return galois::optional<value_type>();
    }
  }
};
GALOIS_WLCOMPILECHECK(TbbPriQueue)
#else
template<typename T = int>
class TbbFIFO;//: public AbstractWorkList<T,false> { };

template<class Compare = std::less<int>, typename T = int>
class PTbb;//: public AbstractWorkList<T,false> { };

template<class Compare = std::less<int>, typename T = int>
class STbb;//: public AbstractWorkList<T,false> { };

template<class Compare = std::less<int>, typename T = int>
class TbbPriQueue;//: public AbstractWorkList<T,false> { };
#endif //TBB

namespace Alt {

class ChunkHeader {
public:
  ChunkHeader* next;
  ChunkHeader() :next(0) {}
};

template<typename T, int chunksize>
class Chunk : public ChunkHeader {
  T data[chunksize];
  int num;
public:
  Chunk() :num(0) {}
  galois::optional<T> pop() {
    if (num)
      return galois::optional<T>(data[--num]);
    else
      return galois::optional<T>();
  }
  bool push(T val) {
    if (num < chunksize) {
      data[num++] = val;
      return true;
    }
    return false;
  }
  template<typename Iter>
  Iter push(Iter b, Iter e) {
    while (b != e && num < chunksize)
      data[num++] = *b++;
    return b;
  }
  bool empty() const {
    return num == 0;
  }
  bool full() const {
    return num == chunksize;
  }
};

class LIFO_SB : private boost::noncopyable {
  substrate::PtrLock<ChunkHeader> head;

public:

  bool empty() const {
    return !head.getValue();
  }

  void push(ChunkHeader* val) {
    ChunkHeader* oldhead = 0;
    do {
      oldhead = head.getValue();
      val->next = oldhead;
    } while (!head.CAS(oldhead, val));
  }

  void pushi(ChunkHeader* val) {
    push(val);
  }

  ChunkHeader* pop() {
    //lock free Fast path (empty)
    if (empty()) return 0;

    //Disable CAS
    head.lock();
    ChunkHeader* C = head.getValue();
    if (!C) {
      head.unlock();
      return 0;
    }
    head.unlock_and_set(C->next);
    C->next = 0;
    return C;
  }

  //returns a chain
  ChunkHeader* steal(LIFO_SB& victim) {
    //lock free Fast path (empty)
    if (victim.empty()) return 0;

    //Disable CAS
    if (!victim.head.try_lock())
      return 0;
    ChunkHeader* C = victim.head.getValue();
    if (!C) {
      victim.head.unlock();
      return 0;
    }
    victim.head.unlock_and_set(C->next);
    C->next = 0;
    return C;
  }
};

class LevelLocalAlt : private boost::noncopyable {
  substrate::PerPackageStorage<LIFO_SB> local;

public:
  void push(ChunkHeader* val) {
    local.getLocal()->push(val);
  }

  void pushi(ChunkHeader* val) {
    push(val);
  }

  ChunkHeader* pop() {
    return local.getLocal()->pop();
  }
};

class LevelStealingAlt : private boost::noncopyable {
  substrate::PerPackageStorage<LIFO_SB> local;

public:
  void push(ChunkHeader* val) {
    local.getLocal()->push(val);
  }

  void pushi(ChunkHeader* val) {
    push(val);
  }

  ChunkHeader* pop() {
    LIFO_SB& me = *local.getLocal();

    ChunkHeader* ret = me.pop();
    if (ret)
      return ret;

    //steal
    int id = substrate::ThreadPool::getPackage();
    for (int i = 0; i < (int) local.size(); ++i) {
      ++id;
      id %= local.size();
      ret = me.steal(*local.getRemote(id));
      if (ret)
	break;
    }
      // if (id) {
      // 	--id;
      // 	id %= local.size();
      // 	ret = me.steal(local.get(id));
      // }
      //      myLock.unlock();
    return ret;
  }
};

template<typename InitWl, typename RunningWl>
class InitialQueue : private boost::noncopyable {
  InitWl global;
  RunningWl local;
public:
  void push(ChunkHeader* val) {
    local.push(val);
  }

  void pushi(ChunkHeader* val) {
    global.pushi(val);
  }

  ChunkHeader* pop() {
    ChunkHeader* ret = local.pop();
    if (ret)
      return ret;
    return global.pop();
  }
};


template<typename gWL = LIFO_SB, int chunksize = 64, typename T = int>
class ChunkedAdaptor : private boost::noncopyable {
  typedef Chunk<T, chunksize> ChunkTy;

  runtime::FixedSizeHeap heap;

  substrate::PerThreadStorage<ChunkTy*> data;

  gWL worklist;

  ChunkTy* mkChunk() {
    return new (heap.allocate(sizeof(ChunkTy))) ChunkTy();
  }

  void delChunk(ChunkTy* C) {
    C->~ChunkTy();
    heap.deallocate(C);
  }

public:
  template<typename Tnew>
  using retype = ChunkedAdaptor<gWL, chunksize, Tnew>;

  typedef T value_type;

  ChunkedAdaptor() : heap(sizeof(ChunkTy)) {
    for (unsigned int i = 0; i  < data.size(); ++i)
      data.get(i) = 0;
  }

  void push(value_type val) {
    ChunkTy*& n = data.get();
    //Simple case, space in current chunk
    if (n && n->push(val))
      return;
    //full chunk, push
    if (n)
      worklist.push(static_cast<ChunkHeader*>(n));
    //get empty chunk;
    n = mkChunk();
    //There better be something in the new chunk
    n->push(val);
  }

  template<typename Iter>
  void push(Iter b, Iter e) {
    ChunkTy*& n = data.get();
    while (b != e) {
      if (!n)
	n = mkChunk();
      b = n->push(b, e);
      if (b != e) {
	worklist.push(static_cast<ChunkHeader*>(n));
	n = 0;
      }
    }
  }

  template<typename RangeTy>
  void push_initial(RangeTy range) {
    typedef typename RangeTy::local_iterator iterator;
    iterator b = range.local_begin();
    iterator e = range.local_end();

    while (b != e) {
      ChunkTy* n = mkChunk();
      b = n->push(b,e);
      worklist.pushi(static_cast<ChunkHeader*>(n));
    }
  }

  galois::optional<value_type> pop() {
    ChunkTy*& n = data.get();
    galois::optional<value_type> retval;
    //simple case, things in current chunk
    if (n && (retval = n->pop()))
      return retval;
    //empty chunk, trash it
    if (n)
      delChunk(n);
    //get a new chunk
    n = static_cast<ChunkTy*>(worklist.pop());
    if (n) {
      return n->pop();
    } else {
      return galois::optional<value_type>();
    }
  }
};
//GALOIS_WLCOMPILECHECK(ChunkedAdaptor)

}

}
}

#endif
