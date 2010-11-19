#ifndef _EMPTYHOARDMANAGER_H_
#define _EMPTYHOARDMANAGER_H_

#include "basehoardmanager.h"
#include "sassert.h"

template <class SuperblockType_>
class EmptyHoardManager : public BaseHoardManager<SuperblockType_> {
public:

  EmptyHoardManager (void)
    : _magic (0x1d2d3d40)
    {}

  int isValid (void) const {
    return (_magic == 0x1d2d3d40);
  }

  typedef SuperblockType_ SuperblockType;

  void free (void *) { abort(); }
  void lock (void) {}
  void unlock (void) {}

  SuperblockType * get (size_t, EmptyHoardManager *) { abort(); return NULL; }
  void put (SuperblockType *, size_t) { abort(); }

private:

  unsigned long _magic;

};


#endif
