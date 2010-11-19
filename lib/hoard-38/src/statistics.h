#ifndef _STATISTICS_H_
#define _STATISTICS_H_

#include <cassert>

namespace Hoard {

class Statistics {
public:
  Statistics (void)
    : _inUse (0),
      _allocated (0)
  {}
  
  inline int getInUse (void) const { assert (_inUse >= 0); return _inUse; }
  inline int getAllocated (void) const { assert (_allocated >= 0); return _allocated; }
  inline void setInUse (int u) { assert (u >= 0); assert (_inUse >= 0); _inUse = u; }
  inline void setAllocated (int a) { assert (a >= 0); assert (_allocated >= 0); _allocated = a; }
  
private:
  
  /// The number of objects in use.
  int _inUse;
  
  /// The number of objects allocated.
  int _allocated;
};

}

#endif
