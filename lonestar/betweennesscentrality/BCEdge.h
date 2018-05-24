#ifndef _ED_H_
#define _ED_H_

#include "BCNode.h"
#include "control.h"

struct BCEdge {
  using NodeType = BCNode<BC_USE_MARKING, BC_CONCURRENT>;
  NodeType* src;
  NodeType* dst;
  ShortPathType val;
  unsigned level;
  
  BCEdge(NodeType* _src, NodeType* _dst) 
    : src(_src), dst(_dst), val(0), level(infinity) { }
  BCEdge() : src(0), dst(0), val(0), level(infinity) { }

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
		if (level != infinity) {
			level = infinity;
		}
  }

  void checkClear(int j) {
	  if (level != infinity) {
      galois::gError(j, " PROBLEM WITH LEVEL OF ", toString()); 
	  }
	  if (val != 0) {
      galois::gError(j, " PROBLEM WITH VAL OF ", toString()); 
	  }
  }

  /**
   * TODO actually implement this if needed
   */
  //char isAlreadyIn() {
  //  return 0;
  //}

  std::string toString() const {
    std::ostringstream s;
    s << val << " " << level;
    return s.str();
  }
};
#endif
