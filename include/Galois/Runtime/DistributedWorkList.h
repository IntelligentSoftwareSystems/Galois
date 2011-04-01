// "Distributed" Shared Memory Worklists -*- C++ -*-

#ifndef __DISTRIBUTEDWORKLIST_H_
#define __DISTRIBUTEDWORKLIST_H_

#include <algorithm>
#include <tr1/unordered_map>

namespace GaloisRuntime {
namespace WorkList {

template<typename T, typename LocalWL, typename GlobalWL, typename DistPolicy>
class RequestHirarchy {
public:
  typedef T value_type;

private:
  typedef cache_line_storage<GlobalWL> paddedSharedWL;
  
  PerCPU<typename LocalWL::template rethread<false>::WL> localQueues;
  paddedSharedWL* sharedQueues;
  unsigned long* starvingFlags;

  int getIndexByLevel(int ID, int level) {
    assert(level > 0);
    int count = 0;
    //skip over some levels
    for (int i = 1; i < level; ++i)
      count += DistPolicy::getNumInLevel(i);
    //index into current level
    count += DistPolicy::mapInto(ID, level);
    return count;
  }

  paddedSharedWL& getSharedByLevel(int ID, int level) {
    return sharedQueues[getIndexByLevel(ID,level)];
  }

  unsigned long& getStarvingByLevel(int ID, int level) {
    return starvingFlags[getIndexByLevel(ID, level)];
  }

  std::pair<bool, value_type> popByLevel(int ID, int level) {
    if (level == 0)
      return localQueues.get(ID).pop();
    return getSharedByLevel(ID, level).data.pop();
  }

  bool pushByLevel(int ID, int level, value_type val) {
    if (level == 0)
      return localQueues.get(ID).push(val);
    return getSharedByLevel(ID, level).data.push(val);
  }

  bool emptyByLevel(int ID, int level) {
    if (level == 0)
      return localQueues.get(ID).empty();
    return getSharedByLevel(ID, level).data.empty();
  }

  int getBitIndex(int ID, int level) {
    return ID % DistPolicy::getNumInBin(level);
  }

  //Set the starving flag for all levels in which the thread
  //is a master thread of a level below
  void setStarving(int ID) {
    for (int i = 1; i < DistPolicy::getNumLevels(); ++i)
      if (DistPolicy::isMasterAtLevel(ID, i - 1)) {
	int index = getBitIndex(ID, i);
	unsigned long& L = getStarvingByLevel(ID, i);
	if (!(L & (0x01UL << index)))
	  __sync_fetch_and_or(&L, 0x01UL << index);
      }
  }

  //Clear the starving flag for all levels in which the thread
  //is a master thread of a level below
  void clearStarving(int ID) {
    for (int i = 1; i < DistPolicy::getNumLevels(); ++i)
      if (DistPolicy::isMasterAtLevel(ID, i - 1)) {
	int index = getBitIndex(ID, i);
	unsigned long& L = getStarvingByLevel(ID, i);
	if (L & (0x01UL << index))
	  __sync_fetch_and_and(&L, ~(0x01UL << index));
      }
  }

  int getHighestStarvingLevel(int ID) {
    int ret = 0;
    for (int i = 1; i < DistPolicy::getNumLevels(); ++i)
      if (DistPolicy::isMasterAtLevel(ID, i - 1) &&
	  getStarvingByLevel(ID, i))
	ret = i;
    return ret;
  }

public:
  RequestHirarchy() :localQueues(0) {
    //count shared queues
    int count = 0;
    for (int i = 1; i < DistPolicy::getNumLevels(); ++i)
      count += DistPolicy::getNumInLevel(i);
    //make shared queues
    sharedQueues = new paddedSharedWL[count];
    starvingFlags = new unsigned long[count];
    for (int i = 0; i < count; ++i)
      starvingFlags[i]= 0;
  }

  ~RequestHirarchy() {
    delete[] sharedQueues;
  }

  bool push(value_type val) {
    int ID = DistPolicy::getID();
    int levelpush = getHighestStarvingLevel(ID);
    pushByLevel(ID, levelpush, val);
  }

  bool aborted(value_type val) {
    return push(val);
  }

  std::pair<bool, value_type> pop() {
    int ID = DistPolicy::getID();
    //Try the local queue first
    std::pair<bool, value_type> ret = popByLevel(ID, 0);
    if (ret.first)
      return ret;

    //walk up the levels
    for (int i = 1; i < DistPolicy::getNumLevels(); ++i) {
      //to go up a level, you have to be a master thread for your old level
      if (!DistPolicy::isMasterAtLevel(ID, i - 1))
	return ret;
      ret = popByLevel(ID, i);
      if (ret.first) {
	clearStarving(ID);
	return ret;
      }
    }
    setStarving(ID);
    return ret;
  }

  std::pair<bool, value_type> try_pop() {
    return pop();
  }

  bool empty() {
    int ID = DistPolicy::getID();
    for (int i = 0; i < DistPolicy::getNumLevels(); ++i) {
      bool ret = emptyByLevel(ID, i);
      if (!ret) return false;
    }
    return true;
  }
  
  //! called in sequential mode to seed the worklist
  template<typename iter>
  void fillInitial(iter begin, iter end) {
    getSharedByLevel(0, DistPolicy::getNumLevels() - 1).fillInitial(begin,end);
  }
};


template<typename T, typename LocalWL, typename DistPolicy>
class ReductionWL {

  typedef cache_line_storage<LocalWL> paddedLocalWL;

  paddedLocalWL* WL;

  sFIFO<T> backup;

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

  std::pair<bool, value_type> try_pop() {
    return WL[DistPolicy::getThreadIsland()].data.try_pop();
  }

  bool empty() {
    return WL[DistPolicy::getThreadIsland()].data.empty();
  }
  
  template<typename iter>
  void fillInitial(iter begin, iter end) {
    return WL[DistPolicy::getThreadIsland()].data.fillInitial(begin,end);
  }

};

#if 0
template<class T, class Indexer, typename ContainerTy = FIFO<T>, bool concurrent=true>
class dOrderByIntegerMetric : private boost::noncopyable {

  
  ContainerTy* current;
  TerminationDetection currentEmpty;
  std::tr1::unordered_map<unsigned int, ContainerTy> data;

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

template<typename T, int chunksize=64, bool concurrent=true>
class dChunkedFIFO : private boost::noncopyable {
  class Chunk : public FixedSizeRing<T, chunksize, false> {
  public:
    Chunk() :next(0) {}
    Chunk* next;
  };

  MM::FixedSizeAllocator heap;


  typedef PtrLock<Chunk*, concurrent> LevelItem;

  PerCPU<Chunk*> cur;
  PerLevel<LevelItem > Items;

  void pushChunk(Chunk* C) OPTNOINLINE {
    LevelItem& I = Items.get();
    I.lock();
    C->next = I.getValue();
    I.unlock_and_set(C);
  }

  Chunk* popChunkByID(unsigned int i) OPTNOINLINE {
    LevelItem& I = Items.get(i);
    //fast-path (lock-free) empty case
    if (!I.getValue())
      return 0;

    I.lock();
    Chunk* retval = I.getValue();
    if (retval) {
      I.unlock_and_set(retval->next);
      retval->next = 0;
    } else {
      I.unlock();
    }

    return retval;
  }

  Chunk* popChunk() OPTNOINLINE {
    int id = Items.myEffectiveID();
    Chunk* r = popChunkByID(id);
    if (r)
      return r;

    for (int i = 0; i < Items.size(); ++i) {
      ++id;
      id %= Items.size();
      r = popChunkByID(id);
      if (r)
	return r;
    }
    return 0;
  }

public:
  template<bool newconcurrent>
  struct rethread {
    typedef dChunkedFIFO<T, chunksize, newconcurrent> WL;
  };

  typedef T value_type;
  
  dChunkedFIFO() : heap(sizeof(Chunk)) {
  }

  bool push(value_type val) OPTNOINLINE {
    Chunk*& n = cur.get();
    if (n && n->full()) {
      pushChunk(n);
      n = 0;
    }
    if (!n)
      n = new (heap.allocate(sizeof(Chunk))) Chunk();
    bool retval = n->push_back(val);
    assert(retval);
    return retval;
  }

  std::pair<bool, value_type> pop() OPTNOINLINE {
    Chunk*& n = cur.get();
    if (n) {
      if (n->empty()) {
	n->~Chunk();
	heap.deallocate(n);
	n = 0;
      } else {
	return n->pop_front();
      }
    } else {
      n = popChunk();
      if (n)
	return pop();
      else
	return std::make_pair(false, value_type());
    }
  }
  
  std::pair<bool, value_type> try_pop() {
    return pop();
  }
  
  bool empty() OPTNOINLINE {
    Chunk* n = cur.get();
    if (n && !n->empty()) return false;
    if (Items.get().getValue()) return false;
    return true;
  }

  bool aborted(value_type val) {
    return push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    for( ; ii != ee; ++ii) {
      push(*ii);
    }
  }

};


}
}

#endif
