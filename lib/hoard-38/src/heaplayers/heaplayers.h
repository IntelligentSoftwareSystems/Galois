// -*- C++ -*-

/*

  Heap Layers: An Extensible Memory Allocation Infrastructure
  
  Copyright (C) 2000-2004 by Emery Berger
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

/**
 * @file heaplayers.h
 * @brief The master Heap Layers include file.
 *
 * Heap Layers is an extensible memory allocator infrastructure.  For
 * more information, read the PLDI 2001 paper "Composing
 * High-Performance Memory Allocators", by Emery D. Berger, Benjamin
 * G. Zorn, and Kathryn S. McKinley.
 * (http://citeseer.ist.psu.edu/berger01composing.html)
 */

#ifndef _HEAPLAYERS_H_
#define _HEAPLAYERS_H_

namespace HL {};

#include "hldefines.h"

/**
 * @def ALLOCATION_STATS
 *
 * Defining ALLOCATION_STATS below as 1 enables tracking of allocation
 * statistics in a variety of layers. You then must link in
 * definitions of the extern variables used therein; stats.cpp
 * contains these definitions.
 *
 * This should be undefined for all but experimental use.
 *
 */

#ifndef ALLOCATION_STATS
#define ALLOCATION_STATS 0
#endif



#ifdef _MSC_VER
// 4786: Disable warnings about long (> 255 chars) identifiers.
// 4512: Disable warnings about assignment operators.
#pragma warning( push )
#pragma warning( disable:4786 4512 )
// Set inlining to the maximum possible depth.
#pragma inline_depth(255)
#define inline __forceinline
#endif


#include "util/sassert.h"
// #include "utility.h"		// convenient wrappers to replace C++ new & delete operators
#include "util/dynarray.h"		// an array that grows by doubling
#include "util/myhashmap.h"

// Hiding machine dependencies.

#include "util/cpuinfo.h"
#include "util/timer.h"			// allows high-resolution timing across a wide range of platforms
#include "util/guard.h"
#include "util/fred.h"

// Lock implementations.

#include "spinlock.h"		// spin-then-yield

#if defined(_WIN32)
#include "util/winlock.h"		// critical-sections (i.e., for Windows only)
#include "util/recursivelock.h"	// a wrapper for recursive locking
#endif


// Useful utilities.

#include "unique.h"
#include "tryheap.h"

// Base heaps

#include "zoneheap.h"		// a zone allocator (frees all memory when the heap goes out of scope)

// Adapters

#include "perclassheap.h"	// make a per-class heap


// Freelist-like heaps ("allocation caches")
// NB: All of these should be used for exactly one size class.

#include "freelistheap.h"	// a free list. Never frees memory.
#include "fifofreelist.h"   // a FIFO free list.
#include "fifodlfreelist.h"  // a doubly-linked FIFO free list.
#include "boundedfreelistheap.h"	// a free list with a bounded length. 


#include "nullheap.h"
#include "coalesceheap.h" // A chunk heap with coalescing.
#include "coalesceableheap.h"

// Utility heap layers

#include "sizethreadheap.h"	// Adds size(ptr) & thread(ptr) methods
#include "lockedheap.h"		// Code-locks a heap
#include "checkheap.h"		// Raises assertions if malloc'ed objects aren't right.
// #include "exceptionheap.h"	// Raise an exception if a malloc fails.

#include "sanitycheckheap.h" // Check for multiple frees and mallocs of same locations.
#include "ansiwrapper.h"     // Provide ANSI C like behavior for malloc (alignment, etc.)

// Multi-threaded heaps
//   hashes the thread id across a number of heaps

#include "phothreadheap.h"	// Private-heaps with ownership
#include "threadheap.h"		// Pure-private heaps (sort of)

// Generic heaps

#include "segheap.h"		// A *very general* segregated fits allocator.

// "Standard" heap layers

#include "kingsleyheap.h"	// A power-of-two size class allocator,
							// a la Chris Kingsley's BSD allocator.

// "Top" heaps.
#include "mallocheap.h"		// a thin wrapper around the system's malloc/free
#include "mmapheap.h"		// a wrapper around the system's virtual memory system

// the rest...

#include "oneheap.h"
#include "debugheap.h"
#include "sizeheap.h"
#include "addheap.h"
#include "profileheap.h"
#include "sizeownerheap.h"
#include "hybridheap.h"
#include "traceheap.h"
#include "stlallocator.h"
#include "adapt.h"
#include "util/dllist.h"
#include "dlheap.h"
// #include "logheap.h"
#include "obstackheap.h"
#include "sbrkheap.h"

#include "xallocHeap.h" // 197.parser's heap

#include "staticheap.h"

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif // _HEAPLAYERS_H_
