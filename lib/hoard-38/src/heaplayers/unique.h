/* -*- C++ -*- */

#ifndef _UNIQUEHEAP_H_
#define _UNIQUEHEAP_H_

#include <cstdlib>
#include <new>

/**
 *
 * @class UniqueHeap
 * @brief Instantiates one instance of a class used for every malloc & free.
 * @author Emery Berger
 *
 */

template <class SuperHeap, class Child = int>
class UniqueHeap {
public:

  /**
   * Ensure that the super heap gets created,
   * and add a reference for every instance of unique heap.
   */
  UniqueHeap (void) 
  {
    volatile SuperHeap * forceCreationOfSuperHeap = getSuperHeap();
    addRef();
  }

  /**
   * @brief Delete one reference to the unique heap.
   * When the number of references goes to zero,
   * delete the super heap.
   */
  ~UniqueHeap (void) {
    int r = delRef();
    if (r <= 0) {
      getSuperHeap()->SuperHeap::~SuperHeap();
    }
  }

  // The remaining public methods are just
  // thin wrappers that route calls to the static object.

  inline void * malloc (size_t sz) {
    return getSuperHeap()->malloc (sz);
  }
  
  inline void free (void * ptr) {
    getSuperHeap()->free (ptr);
  }
  
  inline size_t getSize (void * ptr) {
    return getSuperHeap()->getSize (ptr);
  }

  inline int remove (void * ptr) {
    return getSuperHeap()->remove (ptr);
  }

  inline void clear (void) {
    getSuperHeap()->clear();
  }

#if 0
  inline int getAllocated (void) {
    return getSuperHeap()->getAllocated();
  }

  inline int getMaxAllocated (void) {
    return getSuperHeap()->getMaxAllocated();
  }

  inline int getMaxInUse (void) {
    return getSuperHeap()->getMaxInUse();
  }
#endif

private:

  /// Add one reference.
  void addRef (void) {
    getRefs() += 1;
  }

  /// Delete one reference count.
  int delRef (void) {
    getRefs() -= 1;
    return getRefs();
  }

  /// Internal accessor for reference count.
  int& getRefs (void) {
    static int numRefs = 0;
    return numRefs;
  }

  SuperHeap * getSuperHeap (void) {
    static char superHeapBuffer[sizeof(SuperHeap)];
    static SuperHeap * aSuperHeap = (SuperHeap *) (new ((char *) &superHeapBuffer) SuperHeap);
    return aSuperHeap;
  }

  void doNothing (Child *) {}
};


#endif // _UNIQUE_H_
