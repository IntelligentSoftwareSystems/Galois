/*
 * Tuple.h
 *
 *  Created on: Jan 25, 2011
 *      Author: xinsui
 */

#ifndef DTTUPLE_H_
#define DTTUPLE_H_
#include <iostream>
#include <cstdio>
class DTTuple{
	double _x, _y;

public:

	DTTuple(double x, double y){ _x=x; _y=y; }
	DTTuple() {};
	~DTTuple() {};
	inline const double getX() const { return _x; };
	inline const double getY() const { return _y; };
	inline double setX(double x) { _x = x; };
	inline double setY(double y) { _y = y; };
	
	bool operator==(const DTTuple& rhs) const {
      	   if (_x != rhs._x || _y !=rhs._y)
	      return false;
    	   return true;
  	};
  	bool operator!=(const DTTuple& rhs) const {
    	   return !(*this == rhs);
  	};
  	void print(std::ostream& os) const {
    	    char *buf = new char[256];
            sprintf(buf, "(%.4f, %.4f)", _x, _y);
    	    os << buf;
  	};
};

static inline std::ostream& operator<<(std::ostream& os, const DTTuple& rhs) {
  rhs.print(os);
  return os;
}
#endif /* TUPLE_H_ */
