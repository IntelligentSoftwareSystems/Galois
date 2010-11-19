/* -*- C++ -*- */

#ifndef _UTILITY_H_
#define _UTILITY_H_

/**
 * @file utility.h
 * @brief Convenient wrappers that let us new up & delete objects from heaps.
 *
 * 
 * Example:<BR>
 * <TT>
 *
 * Heap h;<BR>
 * <BR> 
 * Foo * foo;<BR>
 * newObject (foo,h);    		// instead of: foo = new Foo;<BR>
 * deleteObject (foo,h);     		// instead of: delete foo;<BR>
 * <BR>
 * Foo * foo2;<BR>
 * newArray (foo2, 10, h); 		// instead of: foo2 = new Foo[10];<BR>
 * deleteArray (foo2, h);     		// instead of: delete [] foo2;<BR>
 *
*/

#include <new.h>


// Construct an object on a given heap.
class newObject {
public:
  template <class Heap, class Object>
  inline void operator() (Object*& o, Heap& h) const {
    o = new (h.malloc (sizeof(Object))) Object;
  }

  template <class Heap, class Object, class A1>
  inline void operator() (Object*& o, const A1& a1, Heap& h) const {
    o = new (h.malloc (sizeof(Object))) Object (a1);
  }

  template <class Heap, class Object, class A1, class A2>
  inline void operator() (Object*& o, const A1& a1, const A2& a2, Heap& h) const {
    o = new (h.malloc (sizeof(Object))) Object (a1, a2);
  }

  template <class Heap, class Object, class A1, class A2, class A3>
  inline void operator() (Object*& o, const A1& a1, const A2& a2, const A3& a3, Heap& h) const {
    o = new (h.malloc (sizeof(Object))) Object (a1, a2, a3);
  }
};


// Delete an object to a given heap.
class deleteObject {
public:
  template <class Heap, class Object>
  inline void operator()(Object*& o, Heap& h) {
    o->~Object();
    h.free (o);
  }
};


class newArray {
public:
  template <class Heap, class Object>
  inline void operator() (Object*& o, unsigned int n, Heap& h) const {
    // Store the number of array elements in the beginning of the space.
    double * ptr = (double *) h.malloc (sizeof(Object) * n + sizeof(double));
    *((unsigned int *) ptr) = n;
    // Initialize every element.
    ptr++;
    Object * ptr2 = (Object *) ptr;
	// Save the pointer to the start of the array.
	o = ptr2;
	// Now iterate and construct every object in place.
    for (unsigned int i = 0; i < n; i++) {
      new ((void *) ptr2) Object;
      ptr2++;
    }
  }
};


class deleteArray {
public:
  template <class Heap, class Object>
  inline void operator()(Object*& o, Heap& h) const {
    unsigned int n = *((unsigned int *) ((double *) o - 1));
    Object * optr = o;
    // Call the destructor on every element in the array.
    for (unsigned int i = 0; i < n; i++) {
      optr->~Object();
      optr++;
    }
	// Free the array.
    h.free ((void *) ((double *) o - 1));
  }
};

#endif
