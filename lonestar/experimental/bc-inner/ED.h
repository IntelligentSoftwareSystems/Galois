#ifndef _ED_H_
#define _ED_H_

#include "BCNode.h"
#include "control.h"

struct ED {
  using ND = BCNode<>;
  ND * src;
  ND * dst;
  double val;
  int level;
#if CONCURRENT
  //volatile char b;
//  char b;
#else
//  char b;
#endif
  
  ED(ND * _src, ND * _dst) : src(_src), dst(_dst), val(0), level(DEF_DISTANCE)/*, b(0)*/ 
  { }
  ED() : src(0), dst(0), val(0), level(DEF_DISTANCE)/*, b(0)*/
  { }

  ED& operator= (ED const& from) {
    if (this != &from) {
      src = from.src;
      dst = from.dst;
      val = from.val;
      level = from.level;
      std::cerr << "Hell";
      //b = from.b;
    }
    return *this;
  }

  inline void reset() {
		if (level != DEF_DISTANCE) {
//			val = 0;
			level = DEF_DISTANCE;
		}
//    b = 0;
  }

  void checkClear(int j) {
	if (level != DEF_DISTANCE) {
		std::cerr << j << " PROBLEM WITH LEVEL OF " << toString() << std::endl;
	}
	if (val != 0) {
		std::cerr << j << " PROBLEM WITH VAL OF " << toString() << std::endl;
	}
/*	if (b != 0) {
		std::cerr << j << " PROBLEM WITH B OF " << toString() << std::endl;
	}*/
  }
  char isAlreadyIn() {
#if CONCURRENT
//    return __sync_fetch_and_or(&b, 1);
/*    char retval = b;
    b = 1;
    return retval;
	*/
    return 0;
#else
    char retval = b;
    b = 1;
    return retval;
#endif
  }

  void markOut() {
#if CONCURRENT
//    __sync_fetch_and_and(&b, 0);
//    b = 0;
#else
//    b = 0;
#endif
  }

  std::string toString() const {
    std::ostringstream s;
    s << src->id << " " << dst->id << " level: " << level << " val: " << val;
    return s.str();
  }
};
#endif
