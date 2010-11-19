// -*- C++ -*-

/*
 * @file   macwrapper.cpp
 * @brief  This file updates malloc etc. to point to replacements.
 * @author Emery Berger <http://www.cs.umass.edu/~emery>
 */

/*

  Heap Layers: An Extensible Memory Allocation Infrastructure
  
  Copyright (c) 1998-2009 Emery Berger, The University of Texas at Austin
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
   
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>

#include <unistd.h>
#include <limits.h>
#include <malloc/malloc.h>
#include <mach/mach.h>

   
extern "C" void _simple_vdprintf(int, const char *, va_list);
   
inline void
nomalloc_printf(const char *format, ...)
{
  va_list ap;
   
  va_start(ap, format);
  _simple_vdprintf(STDOUT_FILENO, format, ap);
  va_end(ap);
}

#ifndef CUSTOM_PREFIX
#define CUSTOM_PREFIX
#endif

#define CUSTOM_GETSIZE(x)    CUSTOM_PREFIX(malloc_usable_size)(x)
#define CUSTOM_MALLOC(x)     CUSTOM_PREFIX(malloc)(x)
#define CUSTOM_FREE(x)       CUSTOM_PREFIX(free)(x)
#define CUSTOM_REALLOC(x,y)  CUSTOM_PREFIX(realloc)(x,y)
#define CUSTOM_CALLOC(x,y)   CUSTOM_PREFIX(calloc)(x,y)
#define CUSTOM_VALLOC(x)     CUSTOM_PREFIX(valloc)(x)

extern "C" {
  size_t CUSTOM_GETSIZE(const void *);
  void * CUSTOM_MALLOC(size_t);
  void   CUSTOM_FREE(void *);
  void * CUSTOM_REALLOC(void *, size_t);
  void * CUSTOM_CALLOC(size_t, size_t);
  void * CUSTOM_VALLOC(size_t);

  typedef size_t (*mysize_type)(malloc_zone_t *zone, const void *ptr);
  typedef void *(*mymalloc_type)(malloc_zone_t *zone, size_t size);
  typedef void *(*mycalloc_type)(malloc_zone_t *zone, size_t num_items, size_t size);
  typedef void *(*myvalloc_type)(malloc_zone_t *zone, size_t size);
  typedef void (*myfree_type)(malloc_zone_t *zone, void *ptr);
  typedef void *(*myrealloc_type)(malloc_zone_t *zone, void *ptr, size_t size);
  typedef void (*mydestroy_type)(malloc_zone_t *zone);

  static mysize_type    originalSize;
  static mymalloc_type  originalMalloc;
  static mycalloc_type  originalCalloc;
  static myvalloc_type  originalValloc;
  static myfree_type    originalFree;
  static myrealloc_type originalRealloc;
  static mydestroy_type originalDestroy;

  static malloc_zone_t * theZone = NULL;

  static void my_init_hook (void) __attribute__((constructor));

  size_t mysize (malloc_zone_t * zone, const void * ptr) {
    size_t sz;
    if (zone == theZone) {
      sz = CUSTOM_GETSIZE(ptr);
    } else {
      sz = originalSize (zone, ptr);
    }
    return sz;
  }

  void * mymalloc (malloc_zone_t * zone, size_t size) {
    void * ptr;
    if (zone == theZone) {
      ptr = CUSTOM_MALLOC(size);
    } else {
      ptr = originalMalloc (zone, size);
    }
    return ptr;
  }

  void * mycalloc (malloc_zone_t * zone, size_t num, size_t size) {
    if (zone == theZone) {
      return CUSTOM_CALLOC(num, size);
    } else {
      return originalCalloc (zone, num, size);
    }
  }
  
  void * myvalloc (malloc_zone_t * zone, size_t size) {
    if (zone == theZone) {
      return CUSTOM_VALLOC(size);
    } else {
      return originalValloc (zone, size);
    }
  }

  void myfree (malloc_zone_t * zone, void * ptr) {
    if (zone == theZone) {
      CUSTOM_FREE(ptr);
    } else {
      originalFree (zone, ptr);
    }
  }

  void * myrealloc (malloc_zone_t * zone, void * ptr, size_t size) {
    if (zone == theZone) {
      return CUSTOM_REALLOC(ptr, size);
    } else {
      return originalRealloc (zone, ptr, size);
    }
  }

  void mydestroy (malloc_zone_t * zone) {
    // Do not allow destruction of the default zone.
    if (zone != theZone) {
      originalDestroy (zone);
    }
  }

  // The name of the library, which we will install in the hook, below.
  static char mallocName[] = "Hoard";

  //
  // Redirect the system malloc.
  //
  static void my_init_hook (void) {

    if (theZone == NULL) {

      theZone = malloc_default_zone();
      
      // Store the old hooks.
      originalSize    = theZone->size;
      originalMalloc  = theZone->malloc;
      originalCalloc  = theZone->calloc;
      originalValloc  = theZone->valloc;
      originalFree    = theZone->free;
      originalRealloc = theZone->realloc;
      originalDestroy = theZone->destroy;
     
      // Point the hooks to the replacement functions.
      theZone->size     = mysize;
      theZone->malloc   = mymalloc;
      theZone->calloc   = mycalloc;
      theZone->valloc   = myvalloc;
      theZone->free     = myfree;
      theZone->realloc  = myrealloc;
      theZone->destroy  = mydestroy;
      theZone->zone_name = mallocName;

      //
      // We aren't replacing everything, so NULL away.
      //

      // Trash the batch callback hooks.
      theZone->batch_malloc = NULL;
      theZone->batch_free = NULL;

      // And kill the introspection pointer (whatever that means).
      theZone->introspect = NULL;

      // And now in Snow Leopard, more to NULL out.
      theZone->memalign   = NULL;
      theZone->free_definite_size = NULL;
    }
 
  }

}


