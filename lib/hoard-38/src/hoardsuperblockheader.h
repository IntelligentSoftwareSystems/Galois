// -*- C++ -*-

#ifndef _HOARDSUPERBLOCKHEADER_H_
#define _HOARDSUPERBLOCKHEADER_H_

#include <cstdlib>

namespace Hoard {

template <class LockType,
	  int SuperblockSize,
	  typename HeapType>
class HoardSuperblock;

template <class LockType,
	  int SuperblockSize,
	  typename HeapType>
class HoardSuperblockHeader {
public:

  HoardSuperblockHeader (size_t sz, size_t bufferSize)
    : _magicNumber (MAGIC_NUMBER),
      _objectSize (sz),
      _objectSizeIsPowerOfTwo (!(sz & (sz - 1)) && sz),
      _totalObjects ((int) (bufferSize / sz)),
      _owner (NULL),
      _prev (NULL),
      _next (NULL),
      _reapableObjects (_totalObjects),
      _objectsFree (_totalObjects),
      _start ((char *) (this + 1)),
      _position (_start)
  {
  }
  
  inline void * malloc (void) {
    assert (isValid());
    void * ptr = reapAlloc();
    if (!ptr) {
      ptr = freeListAlloc();
    }
    assert ((size_t) ptr % sizeof(double) == 0);
    return ptr;
  }

  inline void free (void * ptr) {
    assert (isValid());
    _freeList.insert (reinterpret_cast<FreeSLList::Entry *>(ptr));
    _objectsFree++;
    if (_objectsFree == _totalObjects) {
      clear();
    }
  }

  void clear (void) {
    assert (isValid());
    // Clear out the freelist.
    _freeList.clear();
    // All the objects are now free.
    _objectsFree = _totalObjects;
    _reapableObjects = _totalObjects;
    _position = _start;
  }

  /// @brief Returns the actual start of the object.
  INLINE void * normalize (void * ptr) const {
    assert (isValid());
    size_t offset = (size_t) ptr - (size_t) _start;
    void * p;

    // Optimization note: the modulo operation (%) is *really* slow on
    // some architectures (notably x86-64). To reduce its overhead, we
    // optimize for the case when the size request is a power of two,
    // which is often enough to make a difference.

    if (_objectSizeIsPowerOfTwo) {
      p = (void *) ((size_t) ptr - (offset & (_objectSize - 1)));
    } else {
      p = (void *) ((size_t) ptr - (offset % _objectSize));
    }
    return p;
  }


  size_t getSize (void * ptr) const {
    assert (isValid());
    size_t offset = (size_t) ptr - (size_t) _start;
    size_t newSize;
    if (_objectSizeIsPowerOfTwo) {
      newSize = _objectSize - (offset & (_objectSize - 1));
    } else {
      newSize = _objectSize - (offset % _objectSize);
    }
    return newSize;
  }

  size_t getObjectSize (void) const {
    return _objectSize;
  }

  int getTotalObjects (void) const {
    return _totalObjects;
  }

  int getObjectsFree (void) const {
    return _objectsFree;
  }

  HeapType * getOwner (void) const {
    return _owner;
  }

  void setOwner (HeapType * o) {
    _owner = o;
  }

  bool isValid (void) const {
    return (_magicNumber == MAGIC_NUMBER);
  }

  HoardSuperblock<LockType, SuperblockSize, HeapType> * getNext (void) const {
    return _next;
  }

  HoardSuperblock<LockType, SuperblockSize, HeapType> * getPrev (void) const {
    return _prev;
  }

  void setNext (HoardSuperblock<LockType, SuperblockSize, HeapType> * n) {
    _next = n;
  }

  void setPrev (HoardSuperblock<LockType, SuperblockSize, HeapType> * p) {
    _prev = p;
  }

  void lock (void) {
    _theLock.lock();
  }

  void unlock (void) {
    _theLock.unlock();
  }

private:

  MALLOC_FUNCTION INLINE void * reapAlloc (void) {
    assert (isValid());
    assert (_position);
    // Reap mode.
    if (_reapableObjects > 0) {
      char * ptr = _position;
      _position = ptr + _objectSize;
      _reapableObjects--;
      _objectsFree--;
      return ptr;
    } else {
      return NULL;
    }
  }

  MALLOC_FUNCTION INLINE void * freeListAlloc (void) {
    assert (isValid());
    // Freelist mode.
    char * ptr = reinterpret_cast<char *>(_freeList.get());
    if (ptr) {
      assert (_objectsFree >= 1);
      _objectsFree--;
    }
    return ptr;
  }

  enum { MAGIC_NUMBER = 0xcafed00d };

  /// A magic number used to verify validity of this header.
  const size_t _magicNumber;

  /// The object size.
  const size_t _objectSize;

  /// True iff size is a power of two.
  const bool _objectSizeIsPowerOfTwo;

  /// Total objects in the superblock.
  const int _totalObjects;

  /// The lock.
  LockType _theLock;

  /// The owner of this superblock.
  HeapType * _owner;

  /// The preceding superblock in a linked list.
  HoardSuperblock<LockType, SuperblockSize, HeapType> * _prev;

  /// The succeeding superblock in a linked list.
  HoardSuperblock<LockType, SuperblockSize, HeapType> * _next;
    
  /// The number of objects available to be 'reap'ed.
  int _reapableObjects;

  /// The number of objects available for (re)use.
  int _objectsFree;

  /// The start of reap allocation.
  char * _start;

  /// The cursor into the buffer following the header.
  char * _position;

  /// The list of freed objects.
  FreeSLList _freeList;

private:

  // Force alignment.
  union {
    char _dchar;
    short _dshort;
    int _dint;
    long _dlong;
    float _dfloat;
    double _ddouble;
    long double _dldouble;
  };
};

}

#endif
