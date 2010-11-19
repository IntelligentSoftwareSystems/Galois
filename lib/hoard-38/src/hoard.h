// -*- C++ -*-

/*

  Heap Layers: An Extensible Memory Allocation Infrastructure
  
  Copyright (c) 1998-2006 Emery Berger, The University of Texas at Austin
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

#ifndef _HOARD_H_
#define _HOARD_H_

#include "hldefines.h"

// The minimum allocation grain for a given object -
// that is, we carve objects out of chunks of this size.
#define SUPERBLOCK_SIZE 65536

// The number of 'emptiness classes'; see the ASPLOS paper for details.
#define EMPTINESS_CLASSES 8


// Hoard-specific Heap Layers

#include "check.h"
#include "fixedrequestheap.h"
#include "hoardmanager.h"
#include "addheaderheap.h"
#include "threadpoolheap.h"
#include "redirectfree.h"
#include "ignoreinvalidfree.h"
#include "conformantheap.h"
#include "hoardsuperblock.h"
#include "lockmallocheap.h"
#include "alignedsuperblockheap.h"
#include "alignedmmap.h"
#include "globalheap.h"

// Generic Heap Layers

#include "ansiwrapper.h"
#include "debugheap.h"
#include "lockedheap.h"
#include "winlock.h"
#include "bins4k.h"
#include "bins8k.h"
#include "bins16k.h"
#include "bins64k.h"
#include "oneheap.h"
#include "freelistheap.h"
#include "threadheap.h"
#include "hybridheap.h"
#include "posixlock.h"
#include "spinlock.h"

// Note: I plan to eventually eliminate the use of the spin lock,
// since the right place to do locking is in an OS-supplied library,
// and platforms have substantially improved the efficiency of these
// primitives.

#if defined(_WIN32)
typedef HL::WinLockType TheLockType;
#elif defined(__APPLE__) || defined(__SVR4)
// NOTE: On older versions of the Mac OS, Hoard CANNOT use Posix locks,
// since they may call malloc themselves. However, as of Snow Leopard,
// that problem seems to have gone away.
// typedef HL::PosixLockType TheLockType;
typedef HL::SpinLockType TheLockType;
#else
typedef HL::SpinLockType TheLockType;
#endif

namespace Hoard {

  //
  // There is just one "global" heap, shared by all of the per-process heaps.
  //

  typedef GlobalHeap<SUPERBLOCK_SIZE, EMPTINESS_CLASSES, TheLockType>
  TheGlobalHeap;
  
  //
  // When a thread frees memory and causes a per-process heap to fall
  // below the emptiness threshold given in the function below, it
  // moves a (nearly or completely empty) superblock to the global heap.
  //

  class hoardThresholdFunctionClass {
  public:
    inline static bool function (int u, int a, size_t objSize) {
      /*
	Returns 1 iff we've crossed the emptiness threshold:
	
	U < A - 2S   &&   U < EMPTINESS_CLASSES-1/EMPTINESS_CLASSES * A
	
      */
      bool r = ((EMPTINESS_CLASSES * u) < ((EMPTINESS_CLASSES-1) * a)) && ((u < a - (2 * SUPERBLOCK_SIZE) / (int) objSize));
      return r;
    }
  };
  

  class SmallHeap;
  
  typedef HoardSuperblock<TheLockType, SUPERBLOCK_SIZE, SmallHeap> SmallSuperblockType;

  //
  // The heap that manages small objects.
  //
  class SmallHeap : 
    public ConformantHeap<
    HoardManager<AlignedSuperblockHeap<TheLockType, SUPERBLOCK_SIZE>,
		 TheGlobalHeap,
		 SmallSuperblockType,
		 EMPTINESS_CLASSES,
		 TheLockType,
		 hoardThresholdFunctionClass,
		 SmallHeap> > {};

  class BigHeap;

  typedef HoardSuperblock<TheLockType, SUPERBLOCK_SIZE, BigHeap> BigSuperblockType;

  // The heap that manages large objects.
  class BigHeap :
    public ConformantHeap<HL::LockedHeap<TheLockType,
					 AddHeaderHeap<BigSuperblockType,
						       SUPERBLOCK_SIZE,
						       AlignedMmap<SUPERBLOCK_SIZE, TheLockType> > > >
  {};


  enum { BigObjectSize = 
	 HL::bins<SmallSuperblockType::Header, SUPERBLOCK_SIZE>::BIG_OBJECT };

  //
  // Each thread has its own heap for small objects.
  //
  class PerThreadHoardHeap :
    public RedirectFree<LockMallocHeap<SmallHeap>,
			SmallSuperblockType> {};

  template <int N, int NH>
  class HoardHeap :
    public HL::ANSIWrapper<
    IgnoreInvalidFree<
      HL::HybridHeap<Hoard::BigObjectSize,
		     ThreadPoolHeap<N, NH, Hoard::PerThreadHoardHeap>,
		     Hoard::BigHeap> > >
  {
  public:
    
    enum { BIG_OBJECT = Hoard::BigObjectSize };
    
    HL::sassert<sizeof(Hoard::BigSuperblockType::Header)
      == sizeof(Hoard::SmallSuperblockType::Header)> ensureSameSizeHeaders;

  };

}


#endif
