// "Distributed" Shared Memory Worklists -*- C++ -*-

namespace GaloisRuntime {
namespace WorkList {


struct FaradayPolicy {
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

}
}
