#ifndef WLCOMPILECHECK
#define WLCOMPILECHECK(name) //
#endif

namespace GaloisRuntime {
namespace WorkList {
namespace Alt {

template<typename T = int, bool concurrent = true>
class LIFO_SB : private boost::noncopyable, private LL::PaddedLock<concurrent> {
  std::deque<T> wl;
  LL::CacheLineStorage<int> size;
  std::vector<T> holder; //put this here to avoid reallocations

  using LL::PaddedLock<concurrent>::lock;
  using LL::PaddedLock<concurrent>::try_lock;
  using LL::PaddedLock<concurrent>::unlock;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef LIFO_SB<T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LIFO_SB<Tnew, concurrent> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    lock();
    __sync_fetch_and_add(&size.data, 1);
    wl.push_back(val);
    unlock();
    return true;
  }

  bool pushi(value_type val)  {
    return push(val);
  }

  std::pair<bool, value_type> pop()  {
    if (!size.data)
      return std::make_pair(false, value_type());
    lock();
    if (wl.empty()) {
      unlock();
      return std::make_pair(false, value_type());
    } else {
      __sync_fetch_and_sub(&size.data, 1);
      value_type retval = wl.back();
      wl.pop_back();
      unlock();
      return std::make_pair(true, retval);
    }
  }

  std::pair<bool,value_type> steal(LIFO_SB& victim) {
    std::pair<bool, value_type> retval(false, value_type());
    if (size.data)
      return pop();
    if (!victim.size.data)
      return retval;
    if (&victim == this)
      return retval;
    if (!LL::TryLockPairOrdered(*this, victim))
      return retval;
    if (victim.wl.empty()) {
      LL::UnLockPairOrdered(*this, victim);
      return std::make_pair(false, value_type());
    } else {
      int num = (victim.wl.size() + 1) / 2;
      __sync_fetch_and_sub(&victim.size.data, num);
      holder.reserve(num);
      for (int i = 0; i < num; ++i) {
	holder.push_back(victim.wl.front());
	victim.wl.pop_front();
      }
      __sync_fetch_and_add(&size.data, num - 1);
      for (int i = 1; i < num; ++i)
	wl.push_back(holder[i]);
      retval.first = true;
      retval.second = holder[0];
      holder.clear();
      LL::UnLockPairOrdered(*this, victim);
      return retval;
    }
  }
};
WLCOMPILECHECK(LIFO_SB);

template<typename ContainerTy = LIFO_SB<>, typename T = int >
class LevelStealingAlt : private boost::noncopyable {
  typedef typename ContainerTy::template rethread<true>::WL LWL;

  PerLevel<LWL> local;
  //  PerLevel<LL::SimpleLock<int, true> > steallock;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef LevelStealingAlt<ContainerTy, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef LevelStealingAlt<typename ContainerTy::template retype<Tnew>::WL, Tnew> WL;
  };

  typedef T value_type;
  
  bool push(value_type val) {
    return local.get().push(val);
  }

  bool pushi(value_type val)  {
    return push(val);
  }

  std::pair<bool, value_type> pop() {
    LWL& me = local.get();
    std::pair<bool, value_type> ret = me.pop();
    if (ret.first)
      return ret;
    //    LL::SimpleLock<int, true>& myLock = steallock.get();
    //    if (myLock.try_lock()) {
      //no one else on this package stealing
      int id = local.myEffectiveID();
      for (int i = 0; i < local.size(); ++i) {
	++id;
	id %= local.size();
	ret = me.steal(local.get(id));
	if (ret.first)
	  break;
      }
      //      myLock.unlock();
      return ret;
      //    }
    //else
    //someone else on this package is stealing 
    //so just do nothing until they are done
      //    myLock.lock();
      //    myLock.unlock();
      //    return pop();
  }
};
WLCOMPILECHECK(LevelStealingAlt);


template<typename gWL = LIFO_SB<>, int chunksize = 64, bool isStack = true, typename T = int>
class ChunkedAdaptor : private boost::noncopyable {
  typedef FixedSizeRing<T, chunksize, false> Chunk;

  MM::FixedSizeAllocator heap;

  class p {
    Chunk* cur;
    Chunk* next;
  public:
    Chunk*& getCur() { return cur; }
    Chunk*& getNext() { if (isStack) return cur; else return next; }
    std::pair<bool, T> pop() { 
      if (isStack)
	return cur->pop_back(); 
      else
	return cur->pop_front();
    }
  };

  PerCPU<p> data;

  typename gWL::template retype<Chunk*>::WL worklist;

  Chunk* mkChunk() {
    return new (heap.allocate(sizeof(Chunk))) Chunk();
  }
  
  void delChunk(Chunk* C) {
    C->~Chunk();
    heap.deallocate(C);
  }

public:
  template<typename Tnew>
  struct retype {
    typedef ChunkedAdaptor<gWL, chunksize, isStack, Tnew> WL;
  };
  template<bool newconcurrent>
  struct rethread {
    typedef ChunkedAdaptor WL;
  };

  typedef T value_type;

  ChunkedAdaptor() : heap(sizeof(Chunk)) {
    for (unsigned int i = 0; i  < data.size(); ++i) {
      p& r = data.get(i);
      r.getCur() = 0;
      r.getNext() = 0;
    }
  }

  bool push(value_type val) {
    p& n = data.get();
    //Simple case, space in current chunk
    if (n.getNext() && n.getNext()->push_back(val))
      return true;
    //full chunk, push
    if (n.getNext())
      worklist.push(n.getNext());
    //get empty chunk;
    n.getNext() = mkChunk();
    //There better be something in the new chunk
    bool worked = n.getNext()->push_back(val);
    assert(worked);
    return worked;
  }

  bool pushi(value_type val)  {
    return push(val);
  }

  std::pair<bool, value_type> pop()  {
    p& n = data.get();
    std::pair<bool, value_type> retval;
    //simple case, things in current chunk
    if (n.getCur() && (retval = n.pop()).first)
      return retval;
    //empty chunk, trash it
    if (n.getCur())
      delChunk(n.getCur());
    //get a new chunk
    std::pair<bool, Chunk*> t = worklist.pop();
    if (t.first) {
      n.getCur() = t.second;
      assert(t.second);
      retval = n.pop();
      //new chunk better have something in it
      assert(retval.first);
      return retval;     
    } else {
      n.getCur() = 0;
      return std::make_pair(false, value_type());
    }
  }
};
WLCOMPILECHECK(ChunkedAdaptor);

}
}
}
