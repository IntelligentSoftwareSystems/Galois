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

#ifndef _COMBINEHEAP_H_
#define _COMBINEHEAP_H_


/**
* @class CombineHeap
* @brief Combines MallocHeap and FreeHeap: MallocHeap for malloc, FreeHeap for the rest
*/

namespace HL {

template <class MallocHeap, class FreeHeap>
class CombineHeap : public FreeHeap {

public:

  inline void * malloc (size_t sz) { 
    return mallocheap.malloc (sz); 
  }

  MallocHeap& getMallocHeap (void) {
    return mallocheap;
  }

  inline void clear (void) {
    mallocheap.clear();
    FreeHeap::clear();
  }

  inline void free (void * ptr) {
    printf ("combineheap: free %x, sz = %d!\n", ptr, getSize(ptr));
    FreeHeap::free (ptr);
  }

private:
  MallocHeap mallocheap;

};

};

#endif

