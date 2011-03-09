// "Distributed" Shared Memory Worklists -*- C++ -*-

#include <algorithm>

namespace GaloisRuntime {
namespace WorkList {


struct FaradayPolicy {

  //Total: 0 .. 47
  //Assumed numbering: 0-23 package dense real cores, 24-47 package dense hyper cores

  static const char level[3][48];
  static const bool master[3][48];

  static int getID() {
    return std::max(0, (int)ThreadPool::getMyID() - 1);
  }

  //Hight of hirarchy
  static int getNumLevels() {
    return 3;
  }

  //number of queues in each level
  static int getNumInLevel(int _level) {
    switch (_level) {
    case 0: return 48;
    case 1: return 4;
    case 2: return 1;
    default: return 0;
    };
  }
  //number of threads in each bin at a given level
  static int getNumInBin(int _level) {
    switch (_level) {
    case 0: return 1;
    case 1: return 6;
    case 2: return 48;
    default: return 0;
    };
  }

  static int mapInto(int ID, int _level) {
    return level[_level][ID];
  }

  static bool isMasterAtLevel(int ID, int _level) {
    return master[_level][ID];
  }

  static int getNumIslands() { return 4; }
  static int getThreadIsland() { 
    int i = ThreadPool::getMyID();
    i = std::max(0, i - 1);
    //Round robin
    //return i < getNumIslands();
    //dense
    return i % 6 == 0;
  }
  static bool isThreadMaster() { 
    int i = ThreadPool::getMyID();
    i = std::max(0, i - 1);
    //Round robin
    //i %= 4;
    //dense
    i /= 6;
    return i;
  }
 
};

//FIXME: Not safe for multiple includes
const char FaradayPolicy::level[3][48] = {
  //Level 0: core
  { 0 ,1 ,2 ,3 ,4 ,5 ,
    6 ,7 ,8 ,9 ,10,11,
    12,13,14,15,16,17,
    18,19,20,21,22,23,
    24,25,26,27,28,29,
    30,31,32,33,34,35,
    36,37,38,39,40,41,
    42,43,44,45,49,47},
  //Level 1: package
  {0,0,0,0,0,0,
   1,1,1,1,1,1,
   2,2,2,2,2,2,
   3,3,3,3,3,3,
   0,0,0,0,0,0,
   1,1,1,1,1,1,
   2,2,2,2,2,2,
   3,3,3,3,3,3},
  //Level 2: top level
  {0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0}
};

const bool FaradayPolicy::master[3][48] = {
  //Level0: individual queues
  {1,1,1,1,1,1,
   1,1,1,1,1,1,
   1,1,1,1,1,1,
   1,1,1,1,1,1,
   1,1,1,1,1,1,
   1,1,1,1,1,1,
   1,1,1,1,1,1,
   1,1,1,1,1,1},
  //Level 1: first of each package
  {1,0,0,0,0,0,
   1,0,0,0,0,0,
   1,0,0,0,0,0,
   1,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0},
  //Level 2: one master queue
  {0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0}
};

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


template<class T, class Indexer>
class DistApproxOrderByIntegerMetric : private boost::noncopyable {

  MP_SC_FIFO<T> data[2048];
  
  Indexer I;
  PerCPU<unsigned int> cursor;

  int num() {
    return 2048;
  }

  int active() {
    return getSystemThreadPool().getActiveThreads();
  }

 public:

  typedef T value_type;
  template<bool newconcurrent>
  struct rethread {
    typedef DistApproxOrderByIntegerMetric<T, Indexer> WL;
  };
  template<typename T2>
  struct retype {
    //FIXME: How do you retype an index function
    typedef DistApproxOrderByIntegerMetric<T2, typename Indexer::template retype<T2>::WL> WL;
  };
  
  DistApproxOrderByIntegerMetric(const Indexer& x = Indexer())
    :I(x), cursor(0)
  {
    for (int i = 0; i < active(); ++i)
      cursor.get(i) = i;
  }
  
  bool push(value_type val) {   
    unsigned int index = I(val, std::numeric_limits<unsigned int>::max());
    index %= num();
    assert(index < num());
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


template<typename T, int chunksize=64, bool concurrent=true>
class dChunkedFIFO : private boost::noncopyable {
  class Chunk : public FixedSizeRing<T, chunksize, false> {
  public:
    Chunk* next;
  };

  struct p {
    Chunk* cur;
    Chunk* next;
    p() : cur(0), next(0) {}
    ~p() {
      delete cur;
      delete next;
    }
  };

  PerCPU<p> data;
  PerLevel<Chunk*> Items;

  void pushChunk(Chunk* C) {
    Chunk*& I = Items.get();
    Chunk* oldHead = 0;
    do {
      oldHead = I;
      C->next = I;
    } while(!__sync_bool_compare_and_swap(&I, oldHead, C));
  }

  Chunk* popChunkByID(unsigned int i) {
    Chunk*& I = Items.get(i);
    Chunk* newHead;
    Chunk* retval;
    do {
      retval = I;
      if (!retval) //no items
	return retval;
      newHead = retval->next;
    } while (!__sync_bool_compare_and_swap(&I, retval, newHead));
    retval->next = 0;
    return retval;
  }

  Chunk* popChunk() {
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
  
  dChunkedFIFO() {
    for (int i = 0; i < Items.size(); ++i)
      Items.get(i) = 0;
  }

  bool push(value_type val) {
    p& n = data.get();
    if (n.next && n.next->full()) {
      pushChunk(n.next);
      n.next = 0;
    }
    if (!n.next)
      n.next = new Chunk;
    bool retval = n.next->push_back(val);
    assert(retval);
    return retval;
  }

  std::pair<bool, value_type> pop() {
    p& n = data.get();
    if (n.cur && n.cur->empty()) {
      delete n.cur;
      n.cur = 0;
    }
    if (n.next && !n.next->empty()) {
      return n.next->pop_front();
    }
    if (!n.cur) {
      Chunk* r = popChunk();
      if (r) {
	//Shared queue had data
	n.cur = r;
      } else {
	//Shared queue was empty, check next
	n.cur = n.next;
	n.next = 0;
	if (!n.cur)
	  return std::make_pair(false, value_type());
      }
    }

    return n.cur->pop_front();
  }
  
  std::pair<bool, value_type> try_pop() {
    return pop();
  }
  
  bool empty() {
    p& n = data.get();
    if (n.cur && !n.cur->empty()) return false;
    if (n.next && !n.next->empty()) return false;
    if (Items.get()) return false;
    return true;
  }

  bool aborted(value_type val) {
    return push(val);
  }

  //Not Thread Safe
  template<typename Iter>
  void fill_initial(Iter ii, Iter ee) {
    p& n = data.get();
    for( ; ii != ee; ++ii) {
      push(*ii);
    }
    pushChunk(n.next);
    n.next = 0;
  }

};


}
}
