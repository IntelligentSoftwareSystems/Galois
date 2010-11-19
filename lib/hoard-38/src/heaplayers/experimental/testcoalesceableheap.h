// -*- C++ -*-

// CoalesceableHeap: manages a range of coalesceable memory.

#ifndef _COALESCEABLEHEAP_H_
#define _COALESCEABLEHEAP_H_

#include <assert.h>
#include <new.h>

#define MULTIPLE_HEAP_SUPPORT 0
#define USE_TOP 0

template <class SuperHeap>
class RequireCoalesceable : public SuperHeap {
public:

  // Some thin wrappers over Header methods.
  inline static int getHeap (void * ptr)          { return Header::getHeader(ptr)->getHeap(); }
  inline static void setHeap (void * ptr, int h)  { Header::getHeader(ptr)->setHeap(h); }
  inline static int getPrevHeap (void * ptr)      { return Header::getHeader(ptr)->getPrevHeap(); }
  inline static void setPrevHeap (void * ptr, int h) { Header::getHeader(ptr)->setPrevHeap(h); }

  inline static size_t getSize (void * ptr)       { return Header::getHeader(ptr)->getSize(); }

  inline static void setSize (void * ptr, size_t sz) { Header::getHeader(ptr)->setSize(sz); }
  inline static size_t getPrevSize (void * ptr)   { return Header::getHeader(ptr)->getPrevSize(); }
  inline static void setPrevSize (void * ptr, size_t sz) { Header::getHeader(ptr)->setPrevSize(sz); }
  inline static void markFree (void * ptr)        { Header::getHeader(ptr)->markFree(); }
  inline static void markInUse (void * ptr)       { Header::getHeader(ptr)->markInUse(); }
  inline static void markPrevInUse (void * ptr)   { Header::getHeader(ptr)->markPrevInUse(); }
  inline static void markMmapped (void * ptr)     { Header::getHeader(ptr)->markMmapped(); }
  inline static int isFree (void * ptr)           { return Header::getHeader(ptr)->isFree(); }
  inline static int isPrevFree (void * ptr)       { return Header::getHeader(ptr)->isPrevFree(); }
  inline static int isMmapped (void * ptr)        { return Header::getHeader(ptr)->isMmapped(); }
  inline static void * getNext (void * ptr)       { return Header::getHeader(ptr)->getNext(); }
  inline static void * getPrev (void * ptr)       { return Header::getHeader(ptr)->getPrev(); }


  // The Header for every object, allocated or freed.
  class Header {
  public:

    // Returns the start of the object (i.e., just past the header).
    inline static void * makeObject (void * buf, size_t prevsz, size_t sz) {
      Header * h = new (buf) Header (prevsz, sz);
      // Record this object as in use in the next header.
      h->markInUse();
      // Record this object's size in the next header.
      h->getNextHeader()->setPrevSize (sz);
      return Header::getObject (h);
    }

#if USE_TOP
    // Returns the start of the object, but without updating the next header.
    inline static void * makeTop (void * buf, size_t prevsz, size_t sz) {
      Header * h = new (buf) Header (prevsz, sz);
      // Pretend the previous object is in use to prevent us from
      // accidentally coalescing backwards with non-existent memory.
      h->markPrevInUse();
      h->markInUse();
      return Header::getObject (h);
    }
#endif

    inline void sanityCheck (void) {
#ifndef NDEBUG
      int headerSize = sizeof(Header);
      assert (headerSize <= sizeof(double));
      assert (getSize() == getNextHeader()->getPrevSize());
      assert (isFree() == getNextHeader()->isPrevFree());
      assert (getNextHeader()->getPrev() == getObject(this));
#if 0
      if (isPrevFree()) {
	assert (getPrevSize() == getHeader(getPrev())->getSize());
      }
#endif
#endif
    }

    // Get the header for a given object.
    inline static Header * getHeader (const void * ptr) { return ((Header *) ptr - 1); }

    // Get the object for a given header.
    inline static void * getObject (const Header * hd)  { return (void *) (hd + 1); }

    inline void setPrevSize (const size_t sz){ _prevSize = sz >> SIZE_SHIFT; }
    inline size_t getPrevSize (void) const { return _prevSize << SIZE_SHIFT; }

    inline void markFree (void)        { getNextHeader()->markPrevFree(); }
    inline void markInUse (void)       { getNextHeader()->markPrevInUse(); }
    inline void markMmapped (void)     { setIsMmapped(); } // _isMmapped = IS_MMAPPED; }
    inline void markNotMmapped (void)  { setIsNotMmapped(); } // _isMmapped = NOT_MMAPPED; }
    inline int isFree (void) const     { return getNextHeader()->isPrevFree(); }
    inline int isNextFree (void) const { return getNextHeader()->getNextHeader()->isPrevFree(); }
    inline int isMmapped (void) const  { return getIsMmapped(); } // _isMmapped == IS_MMAPPED; }
    inline void * getPrev (void) const { return ((char *) this) - getPrevSize(); }
    inline void * getNext (void) const { return ((char *) (this + 2)) + getSize(); }

    inline void markPrevFree (void)    { setPrevIsFree(); } // _prevStatus = FREE; }
    inline void markPrevInUse (void)   { setPrevInUse(); } //_prevStatus = IN_USE; }
    inline int isPrevFree (void) const { return getPrevIsFree(); } // return _prevStatus == FREE; }
    inline void setSize (size_t sz) {
      _size = ((sz >> SIZE_SHIFT) << 2) | (_size & 3);
    }

    inline size_t getSize (void) const {
      return (_size >> 2) << SIZE_SHIFT;
    }

	//    inline size_t getSize (void) const { return getSize(); } // return _size << SIZE_SHIFT; }
//    inline void setSize (const size_t sz)    { setSize(sz); } // _size = sz >> SIZE_SHIFT; }

#if MULTIPLE_HEAP_SUPPORT
    inline int getHeap (void) const { return _currHeap; }
    inline void setHeap (int h)     { _currHeap = h; }
    inline int getPrevHeap (void) const { return _prevHeap; }
    inline void setPrevHeap (int h) { _prevHeap = h; }
#else
    inline int getHeap (void) const { return 0; }
    inline void setHeap (int)       {  }
    inline int getPrevHeap (void) const { return 0; }
    inline void setPrevHeap (int)   {  }
#endif


  private:

    inline Header (void) {}
    inline Header (size_t prevsz, size_t sz)
      :
	// Record sizes, shifting to account for "stolen bits".
        _prevSize (prevsz >> SIZE_SHIFT)
#if 0
       ,_size (sz >> SIZE_SHIFT),
	// Assume that objects are NOT mmapped.
	_isMmapped (NOT_MMAPPED)
#if MULTIPLE_HEAP_SUPPORT
	, _prevHeap (0),
	_currHeap (0)
#endif
#endif
      {
			_size = 0;
	setSize (sz);
	setIsNotMmapped();
	assert (sizeof(Header) <= sizeof(double));
      }

    inline Header * getNextHeader (void) const {
      return ((Header *) ((char *) (this + 1) + getSize()));
    }

    // How many bits we can shift size over without losing information.
    //   3 = double-word alignment.
    enum { SIZE_SHIFT = 3 };

#if !(MULTIPLE_HEAP_SUPPORT) // original

    inline int getIsMmapped (void) const {
      return _size & 1;
    }

    inline void setIsMmapped (void) {
      _size |= 1;
    }

    inline void setIsNotMmapped (void) {
      _size &= ~1;
    }

    inline int getPrevIsFree (void) const {
      return _size & 2;
    }

    inline void setPrevIsFree (void) {
      _size |= 2;
    }

    inline void setPrevInUse (void) {
      _size &= ~2;
    }


#if 1
    // The size of the previous object.
    size_t _prevSize;

    // The size of the current object, with NOT_MMAPPED & IN_USE bits.
    size_t _size;
    
#else
    // The size of the previous object.
    enum { NUM_BITS_STOLEN_FROM_PREVSIZE = 0 };
    size_t _prevSize : sizeof(size_t) * 8 - NUM_BITS_STOLEN_FROM_PREVSIZE;

    // The size of the current object.
    enum { NUM_BITS_STOLEN_FROM_SIZE = 2 };
    size_t _size : sizeof(size_t) * 8 - NUM_BITS_STOLEN_FROM_SIZE;
    
    // Is the current object mmapped?
    enum { NOT_MMAPPED = 0, IS_MMAPPED = 1 };
    unsigned int _isMmapped : 1;
    
    // Is the previous object free or in use?
    enum { IN_USE = 0, FREE = 1 };
    unsigned int _prevStatus : 1;
#endif

#else // new support for scalability...

    // Support for 2^5 = 32 heaps.
    enum { NUM_BITS_FOR_HEAP = 5 };

    enum { NUM_BITS_STOLEN_FROM_SIZE = NUM_BITS_FOR_HEAP + 1 };     // 1 for isMmapped
    enum { NUM_BITS_STOLEN_FROM_PREVSIZE = NUM_BITS_FOR_HEAP + 1 }; // 1 for isPrevFree

    // Max object size.
    enum { MAX_OBJECT_SIZE = 1 << (sizeof(size_t) * 8 + SIZE_SHIFT - NUM_BITS_STOLEN_FROM_SIZE) };

    //// word 1 ////

    // The size of the previous object.
    size_t _prevSize : sizeof(size_t) * 8 - NUM_BITS_STOLEN_FROM_PREVSIZE;

    // What's the previous heap?
    unsigned int _prevHeap : NUM_BITS_FOR_HEAP;

    // Is the previous object free or in use?
    enum { FREE = 0, IN_USE = 1 };
    unsigned int _prevStatus : 1;

    //// word 2 ////

    // The size of the current object.
    size_t _size : sizeof(size_t) * 8 - NUM_BITS_STOLEN_FROM_SIZE;

    // What's the current heap?
    unsigned int _currHeap : NUM_BITS_FOR_HEAP;

    // Is the current object mmapped?
    enum { NOT_MMAPPED = 0, IS_MMAPPED = 1 };
    unsigned int _isMmapped : 1;

#endif
  };

};



template <class SuperHeap>
class CoalesceableHeap : public RequireCoalesceable<SuperHeap> {
public:

  CoalesceableHeap (void)
#if USE_TOP
	  : top (NULL)
#endif
  { }

  inline void * malloc (size_t sz) {
#if !USE_TOP
	  void * buf = SuperHeap::malloc (sz + sizeof(Header));
	  if (buf == NULL) {
		  return NULL;
	  } else {
		  Header * header = (Header *) buf;

		  //
		  // Assume that everything allocated is NOT mmapped.
		  // It is the responsibility of a child layer
		  // to mark mmapped objects as such.
		  //

		  header->markNotMmapped ();

		  //
		  // Record the size of this object in the current header
		  // and the next.
		  //

		  header->setSize (sz);
		  Header * nextHeader = Header::getHeader (header->getNext());
		  nextHeader->setSize (0);
		  nextHeader->setPrevSize (sz);

		  //
		  // Mark the subsequent "object" as in use in order to prevent
		  // accidental coalescing.
		  //

		  nextHeader->markInUse ();
		  return Header::getObject (header);
	  }
#else

    if ((top == NULL) || (sz + sizeof(Header) > getSize(top))) {
      // There wasn't enough space in top. Get more memory (enough to
      // hold an extra header).
      void * buf = SuperHeap::malloc (sz + 2 * sizeof(Header));
      if (buf == NULL)
	return NULL;
      void * ptr = Header::makeTop (buf, 0, sz);
      // Is this object contiguous with top?
      if ((top != NULL) && (getNext(top) == ptr)) {
	// It is contiguous. Extend top.
	setSize (top, getSize(top) + 2 * sizeof(Header) + sz);

	 // printf ("Extended top by %d\n", getSize(ptr));

      } else {
	// We got some non-contiguous memory.
	// For now, just abandon top.

	 // printf ("Abandoning top.\n");
	setSize (ptr, sz + sizeof(Header));
	top = ptr;
      }
    }
    assert (getSize(top) >= sz + sizeof(Header));
    // Carve out an sz object from top and advance top.
    size_t oldTopSize = getSize(top);
    void * newObject = top;
    setSize (newObject, sz);
    top = getNext (newObject);
    // Set top's new header.
    setSize (top, oldTopSize - sz - sizeof(Header));
    setPrevSize (top, sz);
    // Mark the new top as in use to prevent it from being coalesced.

    markInUse (top);

	assert (getNext(newObject) == top);
    Header::getHeader(newObject)->sanityCheck();
	//printf ("top = %x\n", top);
	assert (!isFree(top));
    return newObject;
#endif
  }
  
  inline void free (void * ptr) {
    assert (isFree(ptr));
#if USE_TOP
    // Try to extend top if this object is right before it.
    if (getNext(ptr) == top) {
      assert (getSize(ptr) == getPrevSize(top));
      // Extend top backwards.
      setSize (ptr, getSize(ptr) + sizeof(Header) + getSize(top));
      top = ptr;
	// printf ("free: top = %x\n", top);
      return;
    }
#endif
    SuperHeap::free (Header::getHeader(ptr));
  }


#if USE_TOP
  inline void clear (void) {
    top = NULL;
    SuperHeap::clear ();
  }
#endif

private:

#if USE_TOP
  // The top object allocated so far.
  void * top;
#endif

};


#endif // _COALESCEABLEHEAP_H_

