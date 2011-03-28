// Debug Worklists Wrappers -*- C++ -*-

#ifndef __DEBUGWORKLIST_H_
#define __DEBUGWORKLIST_H_

#include <fstream>
#include <map>

namespace GaloisRuntime {
namespace WorkList {

template<typename T, typename Indexer, typename realWL>
class WorkListTracker {
  PerCPU<sFIFO<std::pair<unsigned int, unsigned int>, false > > tracking;
  //global clock
  cache_line_storage<unsigned int> clock;
  //master thread counting towards a tick
  cache_line_storage<unsigned int> thread_clock;

  realWL wl;
  Indexer I;

  struct info {
    unsigned int min;
    unsigned int max;
    unsigned int count;
    unsigned int sum;
    info() :min(std::numeric_limits<unsigned int>::max()), max(0), count(0), sum(0) {}
    void add(unsigned int val) {
      if (val < min)
	min = val;
      if (val > max)
	max = val;
      sum += val;
      ++count;
    }
  };
  
public:
  typedef T value_type;

  WorkListTracker()
  {
    clock.data = 0;
    thread_clock.data = 0;
  }

  ~WorkListTracker() {

    static const char* translation[] = {
      "00", "01", "02", "03", "04", "05", "06", "07", "08", "09",
      "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
      "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
      "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
      "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
      "50", "51", "52", "53", "54", "55", "56", "57", "58", "59"
    };

    
    std::map<unsigned int, info> AMap;
    for (int t = 0; t < tracking.size(); ++t) {
      //For each thread
      std::string name = "tracking.";
      name += translation[t];
      name += ".txt";
      std::ofstream file(name.c_str());
      
      std::map<unsigned int, info> LMap;
      while (!tracking.get(t).empty()) {
	std::pair<bool, std::pair<unsigned int, unsigned int> > r = tracking.get(t).pop();
	if (r.first) {
	  LMap[r.second.first].add(r.second.second);
	  AMap[r.second.first].add(r.second.second);
	}
      }

      for (int x = 0; x < clock.data + 1; ++x) {
	info& i = LMap[x];
	if (i.count)
	  file << x << " " 
	       << ((double)i.sum / (double)i.count) << " "
	       << i.min << " "
	       << i.max << " "
	       << i.count << "\n";
      }
    }
    std::ofstream file("tracking.avg.txt");
    for (int x = 0; x < clock.data + 1; ++x) {
      info& i = AMap[x];
      if (i.count)
	file << x << " " 
	     << ((double)i.sum / (double)i.count) << " "
	     << i.min << " "
	     << i.max << " "
	     << i.count << "\n";
    }
  }

  //! push a value onto the queue
  bool push(value_type val) {
    return wl.push(val);
  }

  bool aborted(value_type val) {
    return wl.push(val);
  }

  std::pair<bool, value_type> pop() {
    std::pair<bool, value_type> ret = wl.pop();
    if (!ret.first) return ret;
    unsigned int index = I(ret.second);
    tracking.get().push(std::make_pair(clock.data, index));
    if (tracking.myEffectiveID() == 0) {
      ++thread_clock.data;
      if (thread_clock.data == 1024*10) {
	thread_clock.data = 0;
	__sync_fetch_and_add(&clock.data, 1);
      }
    }
    return ret;
  }

  std::pair<bool, value_type> try_pop() {
    return pop();
  }

  bool empty() {
    return wl.empty();
  }
  
  template<typename iter>
  void fillInitial(iter begin, iter end) {
    return wl.fillInitial(begin, end);
  }
};

template<typename iWL>
class NoInlineFilter {
  iWL wl;

public:
  typedef typename iWL::value_type value_type;
  
  template<bool concurrent>
  struct rethread {
    typedef NoInlineFilter<typename iWL::template rethread<concurrent>::WL> WL;
  };

  //! push a value onto the queue
  bool push(value_type val) __attribute__((noinline)) {
    return wl.push(val);
  }

  bool aborted(value_type val) __attribute__((noinline)) {
    return wl.push(val);
  }

  std::pair<bool, value_type> pop() __attribute__((noinline)) {
    return wl.pop();
  }

  std::pair<bool, value_type> try_pop() __attribute__((noinline)) {
    return wl.try_pop();
  }

  bool empty() __attribute__((noinline)) {
    return wl.empty();
  }
  
  template<typename iter>
  void fillInitial(iter begin, iter end) {
    return wl.fillInitial(begin, end);
  }
};


}
}

#endif
