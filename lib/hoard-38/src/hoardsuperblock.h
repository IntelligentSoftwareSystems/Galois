// -*- C++ -*-

#ifndef _HOARDSUPERBLOCK_H_
#define _HOARDSUPERBLOCK_H_

#include <cassert>
#include <cstdlib>

#include "hldefines.h"
#include "freesllist.h"
#include "hoardsuperblockheader.h"

namespace Hoard {

  template <class LockType,
	    int SuperblockSize,
	    typename HeapType>
  class HoardSuperblock {
  public:

    HoardSuperblock (size_t sz)
      : _header (sz, BufferSize)
    {
      assert (_header.isValid());
    }
    
    /// @brief Find the start of the superblock by bitmasking.
    /// @note  All superblocks <em>must</em> be naturally aligned, and powers of two.
    static inline HoardSuperblock * getSuperblock (void * ptr) {
      return (HoardSuperblock *)
	(((size_t) ptr) & ~((size_t) SuperblockSize-1));
    }

    INLINE size_t getSize (void * ptr) const {
      if (_header.isValid() && inRange (ptr)) {
	return _header.getSize (ptr);
      } else {
	return 0;
      }
    }


    INLINE size_t getObjectSize (void) const {
      if (_header.isValid()) {
	return _header.getObjectSize();
      } else {
	return 0;
      }
    }

    MALLOC_FUNCTION INLINE void * malloc (size_t sz) {
      sz = sz; // avoid warning
      assert (_header.isValid());
      void * ptr = _header.malloc();
      if (ptr) {
	assert (inRange (ptr));
      }
      return ptr;
    }

    INLINE void free (void * ptr) {
      if (_header.isValid() && inRange (ptr)) {
	// Pointer is in range.
	_header.free (ptr);
      } else {
	// Invalid free.
      }
    }
    
    void clear (void) {
      if (_header.isValid())
	_header.clear();
    }
    
    // ----- below here are non-conventional heap methods ----- //
    
    INLINE bool isValidSuperblock (void) const {
      assert (_header.isValid());
      bool b = _header.isValid();
      return b;
    }
    
    INLINE int getTotalObjects (void) const {
      assert (_header.isValid());
      return _header.getTotalObjects();
    }
    
    /// Return the number of free objects in this superblock.
    INLINE int getObjectsFree (void) const {
      assert (_header.isValid());
      assert (_header.getObjectsFree() >= 0);
      assert (_header.getObjectsFree() <= _header.getTotalObjects());
      return _header.getObjectsFree();
    }
    
    inline void lock (void) {
      assert (_header.isValid());
      _header.lock();
    }
    
    inline void unlock (void) {
      assert (_header.isValid());
      _header.unlock();
    }
    
    inline HeapType * getOwner (void) const {
      assert (_header.isValid());
      return _header.getOwner();
    }

    inline void setOwner (HeapType * o) {
      assert (_header.isValid());
      assert (o != NULL);
      _header.setOwner (o);
    }
    
    inline HoardSuperblock * getNext (void) const {
      assert (_header.isValid());
      return _header.getNext();
    }

    inline HoardSuperblock * getPrev (void) const {
      assert (_header.isValid());
      return _header.getPrev();
    }
    
    inline void setNext (HoardSuperblock * f) {
      assert (_header.isValid());
      assert (f != this);
      _header.setNext (f);
    }
    
    inline void setPrev (HoardSuperblock * f) {
      assert (_header.isValid());
      assert (f != this);
      _header.setPrev (f);
    }
    
    INLINE bool inRange (void * ptr) const {
      // Returns true iff the pointer is valid.
      const size_t ptrValue = (size_t) ptr;
      return ((ptrValue >= (size_t) _buf) &&
	      (ptrValue <= (size_t) &_buf[BufferSize]));
    }
    
    INLINE void * normalize (void * ptr) const {
      void * ptr2 = _header.normalize (ptr);
      assert (inRange (ptr));
      assert (inRange (ptr2));
      return ptr2;
    }

    typedef Hoard::HoardSuperblockHeader<LockType, SuperblockSize, HeapType> Header;

  private:
    
    
    // Disable copying and assignment.
    
    HoardSuperblock (const HoardSuperblock&);
    HoardSuperblock& operator=(const HoardSuperblock&);
    
    enum { BufferSize = SuperblockSize - sizeof(Header) };
    
    /// The metadata.
    Header _header;

    
    /// The actual buffer. MUST immediately follow the header!
    char _buf[BufferSize];
  };

}


#endif
