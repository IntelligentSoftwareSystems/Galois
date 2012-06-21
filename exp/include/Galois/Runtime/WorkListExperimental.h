/** Experimental Worklists -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
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
#ifndef GALOIS_RUNTIME_WORKLISTEXPERIMENTAL_H
#define GALOIS_RUNTIME_WORKLISTEXPERIMENTAL_H

#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/WorkListDebug.h"
#include "Galois/Runtime/PerThreadStorage.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Threads.h"
#include "Galois/Runtime/ll/PaddedLock.h"

#include "Galois/Bag.h"
#include "Galois/Queue.h"

#include "llvm/Support/CommandLine.h"

#ifdef GALOIS_USE_TBB
#define TBB_PREVIEW_CONCURRENT_PRIORITY_QUEUE 1
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_priority_queue.h>
#include <tbb/concurrent_queue.h>
#endif

#include <boost/utility.hpp>
#include <boost/optional.hpp>
#include <ostream>

namespace GaloisRuntime {
namespace WorkList {

template<class T=int, bool concurrent = true>
class StaticAssignment : private boost::noncopyable {
  typedef Galois::Bag<T> BagTy;

  struct TLD {
    BagTy bags[2];
    unsigned round;
    TLD(): round(0) { }
  };

  GaloisRuntime::PerThreadStorage<TLD> tlds;

  pthread_barrier_t barrier1;
  pthread_barrier_t barrier2;
  unsigned numActive;
  volatile bool empty;

  //! Redistribute work among threads, returns whether there is any work left
  bool redistribute() {
    // TODO(ddn): avoid copying by implementing Bag::splice(iterator b, iterator e)
    BagTy bag;
    unsigned round = tlds.get().round;
    
    for (unsigned i = 0; i < numActive; ++i) {
      BagTy& o = tlds.get(i).bags[round];
      bag.splice(o);
    }

    size_t total = bag.size();
    size_t blockSize = (total + numActive - 1) / numActive;

    if (!total)
      return false;

    typename BagTy::iterator b = bag.begin();
    typename BagTy::iterator e = b;
    
    if (blockSize < total)
      std::advance(e, blockSize);
    else
      e = bag.end();

    for (size_t i = 0, size = blockSize; i < numActive; ++i, size += blockSize) {
      BagTy& o = tlds.get(i).bags[round];
      std::copy(b, e, std::back_inserter(o));
      b = e;
      if (size + blockSize < total)
        std::advance(e, blockSize);
      else
        e = bag.end();
    }

    return true;
  }

 public:
  typedef T value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef StaticAssignment<T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef StaticAssignment<Tnew,concurrent> WL;
  };

  StaticAssignment(): empty(false) {
    numActive = Galois::getActiveThreads();
    pthread_barrier_init(&barrier1, NULL, numActive);
    pthread_barrier_init(&barrier2, NULL, numActive);
  }

  ~StaticAssignment() {
    pthread_barrier_destroy(&barrier1);
    pthread_barrier_destroy(&barrier2);
  }

  void push(value_type val) {
    TLD& tld = tlds.get();
    tld.bags[(tld.round + 1) & 1].push_back(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
    TLD& tld = tlds.get();
    tld.round = (tld.round + 1) & 1;
  }

  boost::optional<value_type> pop() {
    TLD& tld = tlds.get();
    while (true) {
      if (empty) {
        return boost::optional<value_type>();
      }

      if (!tld.bags[tld.round].empty()) {
        boost::optional<value_type> r(tld.bags[tld.round].back());
        tld.bags[tld.round].pop_back();
        return r;
      }

      pthread_barrier_wait(&barrier1);
      tld.round = (tld.round + 1) & 1;
      if (GaloisRuntime::LL::getTID() == 0) {
        empty = !redistribute();
      }
      pthread_barrier_wait(&barrier2);
    }
  }
};
GALOIS_WLCOMPILECHECK(StaticAssignment)

template<class T, class Indexer = DummyIndexer<T>, typename ContainerTy = FIFO<T>, bool concurrent=true >
class ApproxOrderByIntegerMetric : private boost::noncopyable {
  typename ContainerTy::template rethread<concurrent>::WL data[2048];
  
  Indexer I;
  PerThreadStorage<unsigned int> cursor;

  int num() {
    return 2048;
  }

public:
  typedef T value_type;
  template<bool newconcurrent>
  struct rethread {
    typedef ApproxOrderByIntegerMetric<T, Indexer, ContainerTy, newconcurrent> WL;
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    unsigned int& cur = concurrent ? *cursor.getLocal() : *cursor.getRemote(0);
    boost::optional<value_type> ret = data[cur].pop();
    if (ret)
      return ret;

    //must move cursor
    for (int i = 0; i <= num(); ++i) {
      cur = (cur + 1) % num();
      ret = data[cur].pop();
      if (ret.first)
	return ret;
    }
    return boost::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(ApproxOrderByIntegerMetric)

template<class T, class Indexer = DummyIndexer<T>, typename ContainerTy = FIFO<T>, bool concurrent=true >
class LogOrderByIntegerMetric : private boost::noncopyable {
  typename ContainerTy::template rethread<concurrent>::WL data[sizeof(unsigned int)*8 + 1];
  
  Indexer I;
  PerThreadStorage<unsigned int> cursor;

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
    typedef LogOrderByIntegerMetric<T, Indexer, ContainerTy, newconcurrent> WL;
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    unsigned int& cur = concurrent ? *cursor.getLocal() : *cursor.getRemote(0);
    boost::optional<value_type> ret = data[cur].pop();
    if (ret)
      return ret;

    //must move cursor
    for (cur = 0; cur < num(); ++cur) {
      ret = data[cur].pop();
      if (ret.first)
	return ret;
    }
    cur = 0;
    return boost::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(LogOrderByIntegerMetric)

template<typename T, typename Indexer = DummyIndexer<T>, typename LocalTy = FIFO<T>, typename GlobalTy = FIFO<T> >
class LocalFilter {
  GlobalTy globalQ;

  struct p {
    typename LocalTy::template rethread<false>::WL Q;
    unsigned int current;
  };
  PerThreadStorage<p> localQs;
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
    typedef LocalFilter WL;
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> r = localQs.get().Q.pop();
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

  MM::FixedSizeAllocator heap;

  PerThreadStorage<PtrLock<Chunk*, true> > write_stack;

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

  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fill_initial(iter begin, iter end) {
    while (begin != end)
      push(*begin++);
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
  PerThreadStorage<typename LocalWL::template rethread<false>::WL> localQueues;
  PerLevel<GlobalWL> sharedQueues;
  PerLevel<unsigned long> starvingFlags;
  GlobalWL gwl;
  unsigned long gStarving;

  //Clear the starving flag for all levels in which the thread
  //is a master thread of a level below
  void clearStarving() {
    gStarving = 0;
    starvingFlags.get() = 0;
  }

public:
  void push(value_type val) {
    if (gStarving)
      gwl.push(val);
    else if (starvingFlags.get())
      sharedQueues.push(val);
    else
      localQueues.push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    //Try the local queue first
    boost::optional<value_type> ret = localQueues.get().pop();
    if (ret)
      return ret;

    //check parent
    ret = sharedQueues.get().pop();
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
    starvingFlags.get() = 1;
    //if we are master for the package, handle flags
    if (sharedQueues.isFirstInLevel())

    return ret;
  }
};
GALOIS_WLCOMPILECHECK(RequestHierarchy)

template<typename T, typename LocalWL, typename DistPolicy>
class ReductionWL {
  typedef LL::CacheLineStorage<LocalWL> paddedLocalWL;

  paddedLocalWL* WL;
  FIFO<T> backup;
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    int myIsland = DistPolicy::getThreadIsland();
    boost::optional<value_type> val = WL[myIsland].data.pop();
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
    typedef dOrderByIntegerMetric<T, Indexer, typename ContainerTy::template rethread<newconcurrent>::WL, newconcurrent> WL;
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

  //Not Thread Safe
  //Not ideal
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      push(*ii++);
    }
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    bool Empty;
    T retval = take(Empty);
    return !Empty ? 
      boost::optional<value_type>(retval) :
      boost::optional<value_type>();
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
  PerThreadStorage<ChildWLTy> Items;
  int active;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef PartitionedWL<T, Partitioner, ChildWLTy, newconcurrent> WL;
  };

  typedef T value_type;
  
  PartitionedWL(const Partitioner& p = Partitioner()) :P(p), active(Galois::getActiveThreads()) {
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    boost::optional<value_type> r = Items.get().pop();
    // std::cerr << "{" << Items.myEffectiveID() << "}";
    // if (r.first)
    //   std::cerr << r.first;
    return r;
  }
  
  boost::optional<value_type> try_pop() {
    return pop();
  }
};
GALOIS_WLCOMPILECHECK(PartitionedWL)

template<class Compare = std::less<int>, typename T = int>
class SkipListQueue : private boost::noncopyable {

  Galois::ConcurrentSkipListMap<T,int,Compare> wl;
  int magic;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef SkipListQueue<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef SkipListQueue<Compare, Tnew> WL;
  };

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

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    return wl.pollFirstKey();
  }
};
GALOIS_WLCOMPILECHECK(SkipListQueue)

template<class Compare = std::less<int>, typename T = int>
class FCPairingHeapQueue : private boost::noncopyable {
  Galois::FCPairingHeap<T,Compare> wl;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef FCPairingHeapQueue<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef FCPairingHeapQueue<Compare, Tnew> WL;
  };

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

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    return wl.pollMin();
  }
};
GALOIS_WLCOMPILECHECK(FCPairingHeapQueue)

#ifdef GALOIS_USE_TBB
template<typename T = int>
class TbbFIFO : private boost::noncopyable  {
  tbb::concurrent_bounded_queue<T> wl;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef TbbFIFO<T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef TbbFIFO<Tnew> WL;
  };

  typedef T value_type;

  void push(value_type val) {
    wl.push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      wl.push(*b++);
  }

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      wl.push(*b++);
  }

  boost::optional<value_type> pop() {
    T V = T();
    return wl.try_pop(V) ?
      boost::optional<value_type>(V) :
      boost::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(TbbFIFO)

#endif


 template<class Indexer, typename ContainerTy = GaloisRuntime::WorkList::FIFO<>, typename T = int, bool concurrent = true >
   class SimpleOrderedByIntegerMetric : private boost::noncopyable, private GaloisRuntime::LL::PaddedLock<concurrent> {

   using GaloisRuntime::LL::PaddedLock<concurrent>::lock;
   using GaloisRuntime::LL::PaddedLock<concurrent>::try_lock;
   using GaloisRuntime::LL::PaddedLock<concurrent>::unlock;

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
  struct rethread {
    typedef  SimpleOrderedByIntegerMetric<Indexer,ContainerTy,T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef SimpleOrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>::WL,Tnew,concurrent> WL;
  };
  
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    //Fastpath
    CTy* c = current;
    boost::optional<value_type> retval;
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

#ifdef GALOIS_USE_TBB
template<class Indexer, typename ContainerTy = GaloisRuntime::WorkList::ChunkedLIFO<16>, typename T = int, bool concurrent = true >
class CTOrderedByIntegerMetric : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct perItem {
    CTy* current;
    unsigned int curVersion;
    unsigned int scanStart;
    perItem() :current(NULL), curVersion(0), scanStart(0) {}
  };

  struct mhc {
    static size_t hash(int x) { return x; }
    static bool equal(const int& x, const int& y) { return x == y; }
  };

  typedef tbb::concurrent_hash_map<int, CTy*, mhc> HM;
  HM wl;
  int maxV;

  Indexer I;
  GaloisRuntime::PerThreadStorage<perItem> current;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef  CTOrderedByIntegerMetric<Indexer,ContainerTy,T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef CTOrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>::WL,Tnew,concurrent> WL;
  };

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
    CTy* lC = 0;
    if (wl.count(index)) {
      typename HM::const_accessor a;
      wl.find(a, index);
      lC = a->second;
    } else {
      typename HM::accessor a;
      wl.insert(a, index);
      if (!a->second)
	a->second = new CTy();
      lC = a->second;
    }
    if (index < pI.scanStart)
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    fill_work(*this, b, e);
  }

  boost::optional<value_type> pop() {
    //Find a successful pop
    perItem& pI = *current.getLocal();
    CTy*& C = pI.current;
    boost::optional<value_type> retval;
    if (C && (retval = C->pop()))
      return retval;

    //Failed, find minimum bin
    unsigned myID = LL::getTID();
    bool localLeader = LL::isLeaderForPackage(myID);

#if 0
    unsigned msS = 0;
#else
    unsigned msS = pI.scanStart;
    if (localLeader)
      for (int i = 0; i < (int) Galois::getActiveThreads(); ++i)
	msS = std::min(msS, current.getRemote(i)->scanStart);
    else
      msS = std::min(msS, current.getRemote(LL::getLeaderForThread(myID))->scanStart);
#endif

    for (int i = msS; i <= maxV; ++i) {
      typename HM::const_accessor a;
      if (wl.find(a, i)) {
	pI.curVersion = i;
	pI.scanStart = i;
	C = a->second;
	if (C && (retval = C->pop()))
	  return retval;
      }
    }
    return boost::optional<value_type>();
  }
};
GALOIS_WLCOMPILECHECK(CTOrderedByIntegerMetric)
#endif


template<class Indexer, typename ContainerTy, bool concurrent = true, int binmax = 1024*1024 >
class BarrierOBIM : private boost::noncopyable {
  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  volatile unsigned int current;
  volatile unsigned int pushmax;

  CTy* B;

  Indexer I;

  GaloisRuntime::TerminationDetection term;
  pthread_barrier_t barr1;
  pthread_barrier_t barr2;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef BarrierOBIM<Indexer,ContainerTy,newconcurrent,binmax> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef BarrierOBIM<Indexer,typename ContainerTy::template retype<Tnew>::WL,concurrent, binmax> WL;
  };

  typedef typename CTy::value_type value_type;

  BarrierOBIM(const Indexer& x = Indexer())
    :current(0), pushmax(0), I(x)
  {
    B = new CTy[binmax];
    //std::cerr << "$"<<getSystemThreadPool().getActiveThreads() <<"$";
    pthread_barrier_init(&barr1, NULL, Galois::getActiveThreads());
    pthread_barrier_init(&barr2, NULL, Galois::getActiveThreads());
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
    term.workHappened();
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

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    do {
      if (current > pushmax)
	return boost::optional<value_type>();
      do {
	//Find a successful pop
	boost::optional<value_type> retval = B[current].pop();
	if (retval) {
	  term.workHappened();
	  return retval;
	}
	term.localTermination();
      } while (!term.globalTermination());
      
      pthread_barrier_wait(&barr1);
  
      if (GaloisRuntime::LL::getTID() == 0) {
	//std::cerr << "inc: " << current << "\n";
	term.reset();
	if (current <= pushmax)
	  __sync_fetch_and_add(&current, 1);
      }

      pthread_barrier_wait(&barr2);
    } while (true);
  }
};

template<typename T = int, bool concurrent = true>
  class Random : private boost::noncopyable, private GaloisRuntime::LL::PaddedLock<concurrent>  {
  std::vector<T> wl;
  unsigned int seed;
  using GaloisRuntime::LL::PaddedLock<concurrent>::lock;
  using GaloisRuntime::LL::PaddedLock<concurrent>::try_lock;
  using GaloisRuntime::LL::PaddedLock<concurrent>::unlock;

  unsigned int nextRand() {
    seed = 214013*seed + 2531011; 
    return (seed >> 16) & 0x7FFF;
  }

public:
  Random(): seed(0xDEADBEEF) { }

  template<bool newconcurrent>
  struct rethread {
    typedef Random<T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef Random<Tnew, concurrent> WL;
  };

  typedef T value_type;

  void push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    lock();
    if (wl.empty()) {
      unlock();
      return boost::optional<value_type>();
    } else {
      size_t size = wl.size();
      unsigned int index = nextRand() % size;
      value_type retval = wl[index];
      std::swap(wl[index], wl[size-1]);
      wl.pop_back();
      unlock();
      return boost::optional<value_type>(retval);
    }
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

#ifdef GALOIS_USE_TBB
template<class Compare = std::less<int>, typename T = int>
class PTbb : private boost::noncopyable {
  typedef tbb::concurrent_priority_queue<T,Compare> TBBTy;
  
  struct PTD {
    TBBTy wl;
    GaloisRuntime::LL::PaddedLock<true> L;
    bool V;
    std::vector<T> inq;
  };

  GaloisRuntime::PerThreadStorage<PTD> tld;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef PTbb<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef PTbb<Compare, Tnew> WL;
  };
  
  typedef T value_type;
  
  void push(value_type val) {
    unsigned index = GETID<value_type>()(val) % Galois::getActiveThreads();
    PTD& N = tld.get(index);
    if (index == tld.myEffectiveID())
      N.wl.push(val);
    else {
      N.L.lock();
      N.inq.push_back(val);
      N.V = true;
      N.L.unlock();
    }
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    value_type retval;
    PTD& N = tld.get();
    if (N.V) {
      N.L.lock();
      for (typename std::vector<T>::iterator ii = N.inq.begin(), ee = N.inq.end();
	   ii != ee; ++ii)
	N.wl.push(*ii);
      N.inq.clear();
      N.V = false;
      N.L.unlock();
    }
    if (N.wl.try_pop(retval)) {
      return boost::optional<value_type>(retval);
    } else {
      return boost::optional<value_type>();
    }
  }
};

template<class Compare = std::less<int>, typename T = int>
class TbbPriQueue : private boost::noncopyable {
  tbb::concurrent_priority_queue<T,Compare> wl;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef TbbPriQueue<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef TbbPriQueue<Compare, Tnew> WL;
  };

  typedef T value_type;

  void push(value_type val) {
    wl.push(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    value_type retval;
    if (wl.try_pop(retval)) {
      return boost::optional<value_type>(retval);
    } else {
      return boost::optional<value_type>();
    }
  }
};
GALOIS_WLCOMPILECHECK(TbbPriQueue)
#endif //TBB

template<class T=int, bool concurrent = true>
class StaticPartitioning : private boost::noncopyable {
  typedef Galois::Bag<T> BagTy;

  struct TLD {
    BagTy bags[2];
    unsigned round;
    TLD(): round(0) { }
  };

  GaloisRuntime::PerThreadStorage<TLD> tlds;

  GaloisRuntime::GBarrier barrier1;
  GaloisRuntime::GBarrier barrier2;
  unsigned numActive;
  volatile bool empty;

  //! Redistribute work among threads, returns whether there is any work left
  bool redistribute() {
    // TODO(ddn): avoid copying by implementing Bag::splice(iterator b, iterator e)
    BagTy bag;
    unsigned round = tlds.get().round;
    
    for (unsigned i = 0; i < numActive; ++i) {
      BagTy& o = tlds.get(i).bags[round];
      bag.splice(o);
    }

    size_t total = bag.size();
    size_t blockSize = (total + numActive - 1) / numActive;

    if (!total)
      return false;

    typename BagTy::iterator b = bag.begin();
    typename BagTy::iterator e = b;
    
    if (blockSize < total)
      std::advance(e, blockSize);
    else
      e = bag.end();

    for (size_t i = 0, size = blockSize; i < numActive; ++i, size += blockSize) {
      BagTy& o = tlds.get(i).bags[round];
      std::copy(b, e, std::back_inserter(o));
      b = e;
      if (size + blockSize < total)
        std::advance(e, blockSize);
      else
        e = bag.end();
    }

    return true;
  }

 public:
  typedef T value_type;

  template<bool newconcurrent>
  struct rethread {
    typedef StaticPartitioning<T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef StaticPartitioning<Tnew,concurrent> WL;
  };

  StaticPartitioning(): empty(false) {
    numActive = Galois::getActiveThreads();
    barrier1.reinit(numActive);
    barrier2.reinit(numActive);
  }

  void push(value_type val) {
    TLD& tld = tlds.get();
    tld.bags[(tld.round + 1) & 1].push_back(val);
  }

  template<typename ItTy>
  void push(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
  }

  template<typename ItTy>
  void push_initial(ItTy b, ItTy e) {
    while (b != e)
      push(*b++);
    TLD& tld = tlds.get();
    tld.round = (tld.round + 1) & 1;
  }

  boost::optional<value_type> pop() {
    TLD& tld = tlds.get();
    while (true) {
      if (empty) {
        return boost::optional<value_type>();
      }

      if (!tld.bags[tld.round].empty()) {
        boost::optional<value_type> r(tld.bags[tld.round].back());
        tld.bags[tld.round].pop_back();
        return r;
      }

      barrier1.wait();
      tld.round = (tld.round + 1) & 1;
      if (GaloisRuntime::LL::getTID() == 0) {
        empty = !redistribute();
      }
      barrier2.wait();
    }
  }
};
GALOIS_WLCOMPILECHECK(StaticPartitioning)

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
  boost::optional<T> pop() {
    if (num)
      return boost::optional<T>(data[--num]);
    else
      return boost::optional<T>();
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
  LL::PtrLock<ChunkHeader, true> head;

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
  PerLevel<LIFO_SB> local;
  
public:
  void push(ChunkHeader* val) {
    local.get().push(val);
  }

  void pushi(ChunkHeader* val) {
    push(val);
  }

  ChunkHeader* pop() {
    return local.get().pop();
  }
};

class LevelStealingAlt : private boost::noncopyable {
  PerLevel<LIFO_SB> local;
  
public:
  void push(ChunkHeader* val) {
    local.get().push(val);
  }

  void pushi(ChunkHeader* val) {
    push(val);
  }

  ChunkHeader* pop() {
    LIFO_SB& me = local.get();

    ChunkHeader* ret = me.pop();
    if (ret)
      return ret;
    
    //steal
    int id = local.myEffectiveID();
    for (int i = 0; i < (int) local.size(); ++i) {
      ++id;
      id %= local.size();
      ret = me.steal(local.get(id));
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

  MM::FixedSizeAllocator heap;

  PerThreadStorage<ChunkTy*> data;

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
  struct retype {
    typedef ChunkedAdaptor<gWL, chunksize, Tnew> WL;
  };

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

  template<typename Iter>
  void push_initial(Iter b, Iter e)  {
    while (b != e) {
      ChunkTy* n = mkChunk();
      b = n->push(b,e);
      worklist.pushi(static_cast<ChunkHeader*>(n));
    }
  }

  boost::optional<value_type> pop() {
    ChunkTy*& n = data.get();
    boost::optional<value_type> retval;
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
      return boost::optional<value_type>();
    }
  }
};
//GALOIS_WLCOMPILECHECK(ChunkedAdaptor)

}

}
}

#endif
