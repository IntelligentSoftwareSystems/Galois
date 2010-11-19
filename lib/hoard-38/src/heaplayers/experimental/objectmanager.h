// -*- C++ -*-

/*

  Heap Layers: An Extensible Memory Allocation Infrastructure
  
  Copyright (C) 2000-2003 by Emery Berger
  http://www.cs.umass.edu/~emery
  emery@cs.umass.edu
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/

#ifndef _OBJECTMANAGER_H_
#define _OBJECTMANAGER_H_

#include <stdlib.h>  // for size_t

namespace HL {

class EmptyHeap {};

// The default object manager class.


template <class SuperHeap>
class ObjectManagerBase : public SuperHeap {
public:

  // Return the object's size.
  inline static size_t getSize (void * const ptr) {
    return getObject(ptr)->getSize();
  }

protected:

  //// accessors ////

  // Return the object's overhead (i.e., in the header).
  inline static size_t getOverhead (void * const) {
    return sizeof(AllocatedObject);
  }

  // Is this object free?
  inline static bool isFree (void * const ptr) {
    return getObject(ptr)->isFree();
  }


  // Is the previous object free?
  inline static bool isPrevFree (void * const ptr) {
    return getObject(ptr)->isPrevFree();
  }

  // Previous object
  inline static void * getPrev (void * const ptr) {
    return getObject(ptr)->getPrev();
  }


  // Next object
  inline static void * getNext (void * const ptr) {
    return getObject(ptr)->getNext();
  }


  //// mutators ////
	
  // Instantiate an object in a given space.
  inline static void * makeObject (void * buf, size_t prevSize, size_t sz) {
    AllocatedObject * p = new (buf) AllocatedObject (prevSize, sz);
    return (void *) (p + 1);
  }

  inline static size_t getPrevSize (void * const ptr) {
    return getObject(ptr)->getPrevSize();
  }
	
  // Mark this item as free.
  inline static void markFree (void * ptr) {
    getObject(ptr)->markFree();
    getObject(getNext(ptr))->markPrevFree();
  }

  // Mark this item as in use.
  inline static void markInUse (void * ptr) {
    getObject(ptr)->markInUse();
    getObject(getNext(ptr))->markPrevInUse();
  }


private:


  // Change the object's size.
  inline static void setSize (void * ptr, size_t sz) {
    getObject(ptr)->setSize (sz);
  }

  // Change the object's size.
  inline static void setPrevSize (void * ptr, size_t sz) {
    getObject(ptr)->setPrevSize (sz);
  }


  // All objects managed by this object manager
  // are prefaced by a header of type AllocatedObject,
  // which manages the size of the current & previous allocated objects.

  class AllocatedObject {
    friend class ObjectManagerBase<SuperHeap>;
  private:
    inline AllocatedObject (size_t prevSize, size_t sz)
      : _sz (sz)
      , _prevSize (prevSize)
#if 0 //ndef NDEBUG
      , _magic ((double) MAGIC_NUMBER)
#endif
      {
	// Set the prev size of the next object.
	((AllocatedObject *) ((char *) (this + 1) + _sz))->setPrevSize (sz);
	assert (!isFree());
	assert (!isPrevFree());
      }

    inline size_t getSize (void) const {
      assert (isValid());
      return _sz & ~FREE_BIT;
    }
    inline void setSize (size_t sz) {
      assert (sz > 0);
      assert (isValid());
      _sz = sz;
    }
    inline bool isFree (void) const {
      assert (isValid());
      return _sz & FREE_BIT;
    }
    inline bool isPrevFree (void) const {
      assert (isValid());
      return _prevSize & FREE_BIT;
    }
    // Return the previous object (in address order).
    inline void * getPrev (void) const {
      assert (isValid());
      return (void *) ((char *) this - getPrevSize());
    }
    // Return the next object (in address order).
    inline void * getNext (void) const {
      assert (isValid());
      return (void *) ((char *) (this+2) + getSize());
    }
    inline size_t getPrevSize (void) const {
      assert (isValid());
      return _prevSize & ~FREE_BIT;
    }
    inline void markFree (void) {
      assert (isValid());
      markFree(_sz);
    }
    inline void markInUse (void) {
      assert (isValid());
      markInUse(_sz);
    }
    inline void markPrevFree (void) {
      assert (isValid());
      markFree(_prevSize);
    }
    inline void markPrevInUse (void) {
      assert (isValid());
      markInUse(_prevSize);
    }
    inline void setPrevSize (size_t sz) {
      assert (sz > 0);
      assert (isValid());
      _prevSize = sz;
    }
  private:
    enum { FREE_BIT = 1 };
    enum { MAGIC_NUMBER = 0 };

    int isValid (void) const {
#if 0 // ndef NDEBUG
      return (_magic == (double) MAGIC_NUMBER);
#else
      return 1; //((_sz & 6) == 0);
#endif
    }

    inline static void markInUse (size_t& sz) {
      sz &= ~FREE_BIT;
    }

    inline static void markFree (size_t& sz) {
      sz |= FREE_BIT;
    }


    // The size of the previous object (that is, right behind this one).
    size_t _prevSize;

    // We steal the last bit of the size field
    // for free/in use.
    // If the last bit is 1, the object is free.
    // Otherwise, it's in use.
    size_t _sz;

#if 0 //ndef NDEBUG
    double _magic;
#endif
  };

protected:
  inline static AllocatedObject * getObject (void * const ptr) {
    return (AllocatedObject *) ptr - 1;
  }
};



template <class SuperHeap>
class ProvideObjectManager : public ObjectManagerBase<SuperHeap> {
public:
  inline void * malloc (size_t sz) {
    assert (sz > 0);
    void * ptr = SuperHeap::malloc (sz + getOverhead(NULL));
    if (ptr == NULL)
      return NULL;
    void * p = makeObject (ptr, getPrevSize((char *) ptr + getOverhead(NULL)), sz);
    markInUse (p);
    assert (getSize(p) >= sz);
    assert (getPrevSize(getNext(p)) == getSize(p));
    return p;
  }

  inline void free (void * ptr) {
    int ps = getPrevSize(getNext(ptr));
    int sz = getSize(ptr);
    assert (getPrevSize(getNext(ptr)) == getSize(ptr));
    markFree (ptr);
    SuperHeap::free ((void *) getObject(ptr));
  }
};


template <class SuperHeap>
class ObjectManager : public ObjectManagerBase<SuperHeap> {
public:
#if 1
  inline void * malloc (size_t sz) {
    void * p = SuperHeap::malloc (sz);
    if (p != NULL) {
      assert (getSize(p) >= sz);
      //setPrevSize(getNext(p), getSize(p));
      markInUse (p);
    }
    return p;
  }

  inline void free (void * ptr) {
    markFree (ptr);
    SuperHeap::free (ptr);
  }
#endif
};



class ObjectManagerWrapper {
public:
  template <class S>
  class Manager : public ObjectManager<S> {};
};


class ProvideObjectManagerWrapper {
public:
  template <class S>
  class Manager : public ProvideObjectManager<S> {};
};

template <class S>
class NOP : public S {};

class NullObjectManagerWrapper {
public:
  template <class S>
  class Manager : public NOP<S> {};
};


/*

  This ObjectManagerHeap is a tortuous work-around
  for a bug in Visual C++ 6.0 that finally allows us
  to pass ObjectManager wrappers as template arguments.

  This can go away once support is added for nested templates.

  Wrappers look like this:

  class Wrap1 {
  public:
	template <class S>
		class Manager : public ObjectManager<S> {};
  };

  */


#if __SVR4
template <class SuperHeap, class OMWrapper>
class SelectObjectManagerHeap {
public:
  class TheHeap : public OMWrapper::Manager<SuperHeap> {};
};

#else
template <class SuperHeap, class OMWrapper>
class SelectObjectManagerHeap  {
public:
  class ObjectManagerHeapProxy1 : public OMWrapper {
  public:
    template <class A> class Manager {};
  };

  class ObjectManagerHeapProxy2 : public ObjectManagerHeapProxy1 {
  private:
    typedef typename OMWrapper::Manager<SuperHeap> ManageFoo;
  public:
    class ManageMe : public ManageFoo {};
  };

  class TheHeap : public ObjectManagerHeapProxy2::ManageMe {};
};
#endif

};

#endif

