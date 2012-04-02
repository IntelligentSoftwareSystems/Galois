// MCS queuing spin lock -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef _QUEUING_LOCK_H
#define _QUEUING_LOCK_H

#include <cassert>

namespace GaloisRuntime {

template<bool isALock>
class QueuingLock;
template<>
class QueuingLock<true> {
  PerCPU_ring<int> flags;
  int queuelast;
  int myplaceL;

public:
  QueuingLock() : flags(0), queuelast(0) {
    flags.get(0) = 1;
    for (int i = 1; i < flags.size(); ++i) {
      flags.get(i) = 0;
    }
  }
  
  void lock()
  {
    int myplace = __sync_fetch_and_add(&queuelast, 1); // get ticket
    while (!flags.get(myplace % ThreadPool::getActiveThreads())) {};
    //now in CS
    myplaceL = myplace;
  }
  void unlock() {
    flags.get(myplaceL % ThreadPool::getActiveThreads()) = 0;
    flags.get((myplaceL + 1) % ThreadPool::getActiveThreads()) = 1;
  }
};

// template<>
// class QueuingLock<true> {
//   //K42 MCS queuing spinlock

//   struct Element {
//     Element * nextThread;           // guy blocked on me
//     union {
//       uintptr_t waiter;                // 1 while waiting
//       Element* tail;              // tail of blocked queue
//     };
//   };

//   Element _lock;

//   bool CompareAndStoreSynced(volatile uintptr_t* ptr,
// 				   uintptr_t oldval,
// 				   uintptr_t newval) {
//     bool retval;
//     //    __sync_synchronized();
//     __asm__ __volatile__ ("# KillMemory" : : : "memory");
//     retval = __sync_bool_compare_and_swap(ptr, oldval, newval);
//     __asm__ __volatile__ ("# KillMemory" : : : "memory");
//     //    __sync_synchronized();
//     return retval;
//   }

//   template<typename T>
//   T FetchAndNop(volatile T* ptr) {
//     return *ptr;
//   }

// public:
//   QueuingLock() {}
  
//   void lock()
//   {
//     Element waiterel;
//     Element* myel=&waiterel;
//     Element* el;
    
//     while (1) {
//       el=_lock.tail;
//       if (el==0) {
// 	//Lock not held
// 	if (CompareAndStoreSynced((uintptr_t *)(&_lock.tail),
// 				  0, (uintptr_t)(&_lock))) {
// 	  // got the lock, return
// 	  return;
// 	}
// 	//Try again, something changed
//       } else {
// 	// lock is held
// 	// queue on lock by first making myself tail
// 	// and then pointing previous tail at me
	
// 	myel->nextThread = 0;
// 	myel->waiter = 1;
// 	if (CompareAndStoreSynced((uintptr_t *)(&_lock.tail),
// 				  (uintptr_t)el, (uintptr_t)(myel))) {
// 	  // queued on the lock - now complete chain
// 	  el->nextThread = myel;
// 	  while (FetchAndNop(&(myel->waiter)) != 0) {
// 	  }
// 	  // at this point, I have the lock.  lock.tail
// 	  // points to me if there are no other waiters
// 	  // lock.nextThread is not in use
// 	  _lock.nextThread = 0;
// 	  // CompareAndStore "bets" that there are no
// 	  // waiters.  If it succeeds, the lock is put into
// 	  // the held, nowaiters state.
// 	  if (!CompareAndStoreSynced((uintptr_t *)(&_lock.tail),
// 				     (uintptr_t)(myel), (uintptr_t)(&_lock))) {
// 	    // failed to convert lock back to held/no waiters
// 	    // because there is another waiter
// 	    // spin on my nextThread - it may not be updated yet
// 	    while (FetchAndNop((uintptr_t*)&(myel->nextThread)) == 0) {
// 	    }
// 	    // record head of waiter list in lock, thus
// 	    // eliminating further need for myel
// 	    _lock.nextThread =
// 	      (Element *) FetchAndNop((uintptr_t*)&(myel->nextThread));
// 	  }
// 	  // lock is held by me
// 	  return;
// 	}
// 	//Try again, something changed
//       }
//     }
//   }

//   void unlock()
//   {
//     Element* el;
//     // CompareAndStore betting there are no waiters
//     // if it succeeds, the lock is placed back in the free state
//     if (!CompareAndStoreSynced((uintptr_t*)(&_lock.tail), (uintptr_t)(&_lock), 0)) {
//       // there is a waiter - but nextThread may not be updated yet
//       while ((el=(Element*)FetchAndNop((uintptr_t*)&_lock.nextThread)) == 0) {
//       }
//       el->waiter = 0;
//       // waiter is responsible for completeing the lock state transition
//     }
//   }
// };

// template<>
// class QueuingLock<true> {

//   struct qnode {
//     qnode* next;
//     bool locked;
//     qnode() : next(0), locked(false) {}
//   };

//   PerCPU<qnode> _I;
//   qnode* _lock;

//   //No good and portable atomic xchg
//   qnode* atomic_xchg(qnode** ptr, qnode* val) {
//     do {
//       qnode* oldval = *ptr;
//       if (__sync_bool_compare_and_swap(ptr, oldval, val))
// 	return oldval;
//     } while (true);
//   }

// public:
//   QueuingLock() : _I(0), _lock(0) {}

//   void lock() {
//     qnode* I = &_I.get();
//     I->next = 0;
//     qnode* pred = atomic_xchg(&_lock, I);
//     if (pred) { // queue was empty
//       I->locked = true;
//       pred->next = I;
//       while (I->locked) { //spin
//         Config::mem_pause();
//       };
//     }
//   }

//   void unlock() {
//     qnode* I = &_I.get();
//     if (!I->next) { // no know successor
//       if (__sync_bool_compare_and_swap(&_lock, I, 0))
// 	return;
//       while (!I->next) { // spin
//         Config::mem_pause();
//       }
//     }
//     I->next->locked = false;
//   }
// };

template<>
class QueuingLock<false> {
public:
  void lock() const {}
  void unlock() const {}
};


}

#endif
