/* -*- C++ -*- */

#ifndef _ADDHEAP_H_
#define _ADDHEAP_H_

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

// Reserve space for a class in the head of every allocated object.

#include <assert.h>

namespace HL {

template <class Add, class Super>
class AddHeap : public Super {
public:
  
  inline void * malloc (size_t sz) {
    void * ptr = Super::malloc (sz + align(sizeof(Add)));
    void * newPtr = (void *) align ((size_t) ((Add *) ptr + 1));
    return ptr;
  }
  
  inline void free (void * ptr) {
    void * origPtr = (void *) ((Add *) ptr - 1);
    Super::free (origPtr);
  }

  inline size_t getSize (void * ptr) {
    void * origPtr = (void *) ((Add *) ptr - 1);
    return Super::getSize (origPtr);
  }
  
private:
  static inline size_t align (size_t sz) {
    return (sz + (sizeof(double) - 1)) & ~(sizeof(double) - 1);
  }

};

};
#endif
