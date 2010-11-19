/* -*- C++ -*- */

/**
 * @class ZoneHeap
 * @brief A zone (or arena, or region) based allocator.
 * @author Emery Berger
 * @date June 2000
 *
 * Uses the superclass to obtain large chunks of memory that are only
 * returned when the heap itself is destroyed.
 *
*/

#ifndef _ZONEHEAP_H_
#define _ZONEHEAP_H_

#include <assert.h>

namespace HL {

  template <class Super, size_t ChunkSize>
  class ZoneHeap : public Super {
  public:

    ZoneHeap (void)
      : sizeRemaining (-1),
	currentArena (NULL),
	pastArenas (NULL)
    {}

    ~ZoneHeap (void)
    {
      // printf ("deleting arenas!\n");
      // Delete all of our arenas.
      Arena * ptr = pastArenas;
      while (ptr != NULL) {
	void * oldPtr = (void *) ptr;
	ptr = ptr->nextArena;
	//printf ("deleting %x\n", ptr);
	Super::free (oldPtr);
      }
      if (currentArena != NULL)
	//printf ("deleting %x\n", currentArena);
	Super::free (currentArena);
    }

    inline void * malloc (size_t sz) {
      void * ptr = zoneMalloc (sz);
      return ptr;
    }

    /// Free in a zone allocator is a no-op.
    inline void free (void *) {}

    /// Remove in a zone allocator is a no-op.
    inline int remove (void *) { return 0; }


  private:

    inline static size_t align (int sz) {
      return (sz + (sizeof(double) - 1)) & ~(sizeof(double) - 1);
    }

    inline void * zoneMalloc (size_t sz) {
      void * ptr;
      // Round up size to an aligned value.
      sz = align (sz);
      // Get more space in our arena if there's not enough room in this one.
      if ((currentArena == NULL) || (sizeRemaining < (int) sz)) {
	// First, add this arena to our past arena list.
	if (currentArena != NULL) {
	  currentArena->nextArena = pastArenas;
	  pastArenas = currentArena;
	}
	// Now get more memory.
	size_t allocSize = ChunkSize;
	if (allocSize < sz) {
	  allocSize = sz;
	}
	currentArena =
	  (Arena *) Super::malloc (allocSize + sizeof(Arena));
	if (currentArena == NULL) {
	  return NULL;
	}
	currentArena->arenaSpace = (char *) (currentArena + 1);
	currentArena->nextArena = NULL;
	sizeRemaining = ChunkSize;
      }
      // Bump the pointer and update the amount of memory remaining.
      sizeRemaining -= sz;
      ptr = currentArena->arenaSpace;
      currentArena->arenaSpace += sz;
      assert (ptr != NULL);
      return ptr;
    }
  
    class Arena {
    public:
      Arena * nextArena;
      char * arenaSpace;
      double _dummy; // For alignment.
    };
    
    /// Space left in the current arena.
    long sizeRemaining;

    /// The current arena.
    Arena * currentArena;

    /// A linked list of past arenas.
    Arena * pastArenas;
  };

}

#endif
