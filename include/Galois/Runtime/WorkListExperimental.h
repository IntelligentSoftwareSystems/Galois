// Experimental Shared Memory Worklists -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef __WORKLISTEXPERIMENTAL_H_
#define __WORKLISTEXPERIMENTAL_H_

namespace GaloisRuntime {
namespace WorkList {
namespace Experimental {

template<class T, typename ContainerTy = FIFO<T> >
class StealingLocalWL : private boost::noncopyable {

  PerCPU<ContainerTy> data;

  // static void merge(ContainerTy& x, ContainerTy& y) {
  //   assert(x.empty());
  //   assert(y.empty());
  // }

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef StealingLocalWL<T, ContainerTy> WL;
  };

  typedef T value_type;
  
  StealingLocalWL() {}

  bool push(value_type val) {
    return data.get().push(val);
  }

  std::pair<bool, value_type> pop() {
    std::pair<bool, value_type> ret = data.get().pop();
    if (ret.first)
      return ret;
    return data.getNext().pop();
  }

  bool empty() {
    return data.get().empty();
  }
  bool aborted(value_type val) {
    return push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    while (ii != ee) {
      push(*ii++);
    }
  }
};
WLCOMPILECHECK(StealingLocalWL);

template<class T, class Indexer = DummyIndexer<T>, typename ContainerTy = FIFO<T>, bool concurrent=true >
class ApproxOrderByIntegerMetric : private boost::noncopyable {

  typename ContainerTy::template rethread<concurrent>::WL data[2048];
  
  Indexer I;
  PerCPU<unsigned int> cursor;

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
      cursor.get(i) = 0;
  }
  
  bool push(value_type val) OPTNOINLINE {   
    unsigned int index = I(val);
    index %= num();
    assert(index < num());
    return data[index].push(val);
  }

  std::pair<bool, value_type> pop() OPTNOINLINE {
    // print();
    unsigned int& cur = concurrent ? cursor.get() : cursor.get(0);
    std::pair<bool, value_type> ret = data[cur].pop();
    if (ret.first)
      return ret;

    //must move cursor
    for (int i = 0; i <= num(); ++i) {
      cur = (cur + 1) % num();
      ret = data[cur].pop();
      if (ret.first)
	return ret;
    }
    return std::pair<bool, value_type>(false, value_type());
  }

  bool empty() OPTNOINLINE {
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
WLCOMPILECHECK(ApproxOrderByIntegerMetric);

template<class T, class Indexer = DummyIndexer<T>, typename ContainerTy = FIFO<T>, bool concurrent=true >
class LogOrderByIntegerMetric : private boost::noncopyable {

  typename ContainerTy::template rethread<concurrent>::WL data[sizeof(unsigned int)*8 + 1];
  
  Indexer I;
  PerCPU<unsigned int> cursor;

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
      cursor.get(i) = 0;
  }
  
  bool push(value_type val) {   
    unsigned int index = I(val);
    index = getBin(index);
    return data[index].push(val);
  }

  std::pair<bool, value_type> pop() {
    // print();
    unsigned int& cur = concurrent ? cursor.get() : cursor.get(0);
    std::pair<bool, value_type> ret = data[cur].pop();
    if (ret.first)
      return ret;

    //must move cursor
    for (cur = 0; cur < num(); ++cur) {
      ret = data[cur].pop();
      if (ret.first)
	return ret;
    }
    cur = 0;
    return std::pair<bool, value_type>(false, value_type());
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
WLCOMPILECHECK(LogOrderByIntegerMetric);

template<typename T, typename Indexer = DummyIndexer<T>, typename LocalTy = FIFO<T>, typename GlobalTy = FIFO<T> >
class LocalFilter {
  GlobalTy globalQ;

  struct p {
    typename LocalTy::template rethread<false>::WL Q;
    unsigned int current;
  };
  PerCPU<p> localQs;
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
  bool push(value_type val) OPTNOINLINE {
    unsigned int index = I(val);
    p& me = localQs.get();
    if (index <= me.current)
      return me.Q.push(val);
    else
      return globalQ.push(val);
  }

  //! push an aborted value onto the queue
  bool aborted(value_type val) {
    return push(val);
  }

  //! pop a value from the queue.
  std::pair<bool, value_type> pop() OPTNOINLINE {
    std::pair<bool, value_type> r = localQs.get().Q.pop();
    if (r.first)
      return r;
    
    r = globalQ.pop();
    if (r.first)
      localQs.get().current = I(r.second);
    return r;
  }

  //! return if the queue *may* be empty
  bool empty() OPTNOINLINE {
    if (!localQs.get().Q.empty()) return false;
    return globalQ.empty();
  }
  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fill_initial(iter begin, iter end) {
    globalQ.fill_initial(begin,end);
  }
};
WLCOMPILECHECK(LocalFilter);

#if 0
//Bag per writer, reader steals entire bag
template<typename T, int chunksize = 64>
class MP_SC_Bag {
  class Chunk : public FixedSizeRing<T, chunksize, false>, public ConExtListNode<Chunk>::ListNode {};

  MM::FixedSizeAllocator heap;

  PerCPU<PtrLock<Chunk*, true> > write_stack;

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
WLCOMPILECHECK(MP_SC_Bag);
#endif

//Per CPU and Per Level Queues, with staving flags
template<typename T, typename LocalWL, typename GlobalWL>
class RequestHirarchy {
public:
  typedef T value_type;

private:
  PerCPU<typename LocalWL::template rethread<false>::WL> localQueues;
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
  bool push(value_type val) {
    if (gStarved)
      return gwl.push(val);
    if (starvingFlags.get())
      return sharedQueues.push(val);
    return localQueues.push(val);
  }

  bool aborted(value_type val) {
    return push(val);
  }

  std::pair<bool, value_type> pop() {
    //Try the local queue first
    std::pair<bool, value_type> ret = localQueues.get().pop();
    if (ret.first)
      return ret;

    //check parent
    ret = sharedQueues.get().pop();
    if (ret.first) {
      clearStarving();
      return ret;
    }

    //check global
    ret = gwl.pop();
    if (ret.first) {
      clearStarving();
      return ret;
    }
    
    //Any thread can set the package starving flag
    starvingFlags.get() = 1;
    //if we are master for the package, handle flags
    if (sharedQueues.isFirstInLevel())

    return ret;
  }

  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fill_initial(iter begin, iter end) {
    gwl.fill_initial(begin,end);
  }
};


template<typename T, typename LocalWL, typename DistPolicy>
class ReductionWL {

  typedef cache_line_storage<LocalWL> paddedLocalWL;

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

  bool push(value_type val) {
    if (starving)
      return backup.push(val);
    return WL[DistPolicy::getThreadIsland()].data.push(val);
  }

  bool aborted(value_type val) {
    return WL[DistPolicy::getThreadIsland()].data.aborted(val);
  }

  std::pair<bool, value_type> pop() {
    int myIsland = DistPolicy::getThreadIsland();
    std::pair<bool, value_type> val = WL[myIsland].data.pop();
    if (val.first || !DistPolicy::isThreadMaster())
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

  template<typename iter>
  void fillInitial(iter begin, iter end) {
    return gwl.fill_initial(begin,end);
  }
};


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
    :I(x), numActive(getSystemThreadPool().getActiveThreads())
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
class GWL_Idempotent_FIFO : private boost::noncopyable {

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
  typedef GWL_Idempotent_FIFO<T> ConcurrentTy;
  typedef GWL_Idempotent_FIFO<T> SingleTy;
  enum {MAYSTEAL = true};

  GWL_Idempotent_FIFO(int size = 256)
    :head(0), tail(0)
  {
    tasks = mkArray(size);
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

}
}
}

#endif
