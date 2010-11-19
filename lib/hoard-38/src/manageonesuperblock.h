// -*- C++ -*-

/**
 * @file manageonesuperblock.h
 * @author Emery Berger <http://www.cs.umass.edu/~emery>
 */


#ifndef _MANAGEONESUPERBLOCK_H_
#define _MANAGEONESUPERBLOCK_H_

/**
 * @class  ManageOneSuperblock
 * @brief  A layer that caches exactly one superblock, thus avoiding costly lookups.
 * @author Emery Berger <http://www.cs.umass.edu/~emery>
 */

namespace Hoard {

template <class SuperHeap>
class ManageOneSuperblock : public SuperHeap {
public:

  typedef typename SuperHeap::SuperblockType SuperblockType;

  /// Get memory from the current superblock.
  inline void * malloc (size_t sz) {
    if (_current) {
      void * ptr = _current->malloc (sz);
      if (ptr) {
	assert (_current->getSize(ptr) >= sz);
	return ptr;
      }
    }
    return slowMallocPath (sz);
  }

  /// Try to free the pointer to this superblock first.
  inline void free (void * ptr) {
    SuperblockType * s = SuperHeap::getSuperblock (ptr);
    if (s == _current) {
      _current->free (ptr);
    } else {
      SuperHeap::free (ptr);
    }
  }

  /// Get the current superblock and remove it.
  SuperblockType * get (void) {
    if (_current) {
      SuperblockType * s = _current;
      _current = NULL;
      return s;
    } else {
      // There's none cached, so just get one from the superheap.
      return SuperHeap::get();
    }
  }

  /// Put the superblock into the cache.
  inline void put (SuperblockType * s) {
    if (!s || (s == _current) || (!s->isValidSuperblock())) {
      // Ignore if we already are holding this superblock, of if we
      // got a NULL pointer, or if it's invalid.
      return;
    }
    if (_current) {
      // We have one already -- push it out.
      SuperHeap::put (_current);
    }
    _current = s;
  }

private:

  /// Obtain a superblock and return an object from it.
  void * slowMallocPath (size_t sz) {
    void * ptr = NULL;
    while (!ptr) {
      if (_current) {
	ptr = _current->malloc (sz);
	if (ptr) {
	  return ptr;
	} else {
	  SuperHeap::put (_current);
	}
      }
      _current = SuperHeap::get();
      if (!_current) {
	return NULL;
      }
      ptr = _current->malloc (sz);
    }
    return ptr;
  }

  /// The current superblock.
  SuperblockType * _current;

};

}

#endif
