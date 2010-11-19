/* -*- C++ -*- */

/*

  Heap Layers: An Extensible Memory Allocation Infrastructure
  
  Copyright (C) 2000-2005 by Emery Berger
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

#ifndef _MALLOCHEAP_H_
#define _MALLOCHEAP_H_

#include <cstdlib>

#if defined(__SVR4)
extern "C" size_t malloc_usable_size (void *);
#else
extern "C" size_t malloc_usable_size (void *) throw ();
#endif

#if defined(_WIN32) || defined(linux)
#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

/**
 * @class mallocHeap
 * @brief A "source heap" that uses malloc and free.
 */

namespace HL {

class mallocHeap {
public:

  ~mallocHeap (void) {}
  
  inline void * malloc (size_t sz) {
    return ::malloc (sz);
  }
  
  
  inline void free (void * ptr) {
    ::free (ptr);
  }

#if defined(_MSC_VER)
  inline size_t getSize (void * ptr) {
    return ::_msize (ptr);
  }
#elif defined(__GNUC__) && !defined(__SVR4)
  inline size_t getSize (void * ptr) {
    return ::malloc_usable_size (ptr);
  }
#elif defined(__APPLE__)
  inline size_t getSize (void * ptr) {
    return ::malloc_size (ptr);
  }
#endif
  
};

};

#endif
