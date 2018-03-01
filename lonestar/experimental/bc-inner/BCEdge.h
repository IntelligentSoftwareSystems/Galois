#ifndef _ED_H_
#define _ED_H_

#include "BCNode.h"
#include "control.h"

struct BCEdge {
  BCNode<>* src;
  BCNode<>* dst;
  double val;
  int level;
  
  BCEdge(BCNode<>* _src, BCNode<>* _dst) 
    : src(_src), dst(_dst), val(0), level(DEF_DISTANCE) { }
  BCEdge() : src(0), dst(0), val(0), level(DEF_DISTANCE) { }

  BCEdge& operator= (BCEdge const& from) {
    if (this != &from) {
      src = from.src;
      dst = from.dst;
      val = from.val;
      level = from.level;
    }
    return *this;
  }

  inline void reset() {
		if (level != DEF_DISTANCE) {
			level = DEF_DISTANCE;
		}
  }

  void checkClear(int j) {
	  if (level != DEF_DISTANCE) {
      galois::gError(j, " PROBLEM WITH LEVEL OF ", toString()); 
	  }
	  if (val != 0) {
      galois::gError(j, " PROBLEM WITH VAL OF ", toString()); 
	  }
  }

  /**
   * TODO actually implement this if needed
   */
  char isAlreadyIn() {
    return 0;
  }

  std::string toString() const {
    std::ostringstream s;
    s << src->id << " " << dst->id << " level: " << level << " val: " << val;
    return s.str();
  }
};
#endif
