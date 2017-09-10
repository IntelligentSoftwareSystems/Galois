// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Runtime
//   O|S   :: RunOwnership
//             -> WaitForSerial -> RunSerial -> WaitForOwnership
//

#include "config.h"

#ifdef DMP_ENABLE_MODEL_O_S

#include "dmp-internal.h"
#include "dmp-internal-mot.h"

//--------------------------------------------------------------
// Encoding of MOT[x]:
//   MOT[x] & bit32   :: resource bit
//     When set, x is owned by (grouped by) a resource.
//
//   MOT[x] & bit1-31 :: owner
//     Depends on the kind of ownership:
//     -- Resource :: the resource ID (a hash of the DMPresource pointer)
//     -- Shared   :: DMP_MOT_OWNER_SHARED
//     -- Normal   :: the thread ID of the owner
//--------------------------------------------------------------

#define DMP_MOT_OWNER_RESOURCE_BIT (1<<31)
#define DMP_MOT_OWNER_MASK         ((1ul<<31) - 1)
#define DMP_MOT_OWNER_SHARED       DMP_MOT_OWNER_MASK

static inline uint32_t DMPmotOwnerFromThread(DmpThreadInfo* dmp) {
  return dmp->threadID;
}

static inline uint32_t DMPmotOwnerFromResource(DMPresource* r) {
  // FIXME: this hash-based encoding is broken: two threads could hold
  // two different resources yet both may think they own the same data!
#if defined(__LP_64__) || defined(__LP64__) || defined(__amd64__) || defined(__x64_64__)
  return DMP_MOT_OWNER_RESOURCE_BIT |
         (((uint32_t)(uintptr_t)r >> 3) & DMP_MOT_OWNER_MASK);
#else
  return DMP_MOT_OWNER_RESOURCE_BIT |
         (((uint32_t)(uintptr_t)r >> 2) & DMP_MOT_OWNER_MASK);
#endif
}

//-----------------------------------------------------------------------
// API
//-----------------------------------------------------------------------

void DMP_initRuntime() {
  dmp_static_assert(MaxThreads < DMP_MOT_OWNER_SHARED);
  // LLVM runs out of memory if we use a static initializer for this !!! :-(
  for (int i = 0; i < DMP_MOT_ENTRIES; ++i)
    DMPmot[i] = DMP_MOT_OWNER_SHARED;
}

void DMP_initRuntimeThread() {
}

//-----------------------------------------------------------------------
// MOT Policies
//-----------------------------------------------------------------------

static inline int slowOwnershipCheck(const int owner) {
  // Slow path ownership checks: return true of 'DMPMAP' owns 'owner'.
#ifdef DMP_ENABLE_DATA_GROUPING
  if ((owner & DMP_MOT_OWNER_RESOURCE_BIT) != 0) {
    // Check if 'owner' is a held resource.
    if (owner == DMPmotOwnerFromResource(DMPMAP->innerResource))
      return 1;
    DMPresource* r;
    for (r = DMPMAP->nextResource; r; r = r->outer)
      if (owner == DMPmotOwnerFromResource(r))
        return 1;
  }
#endif
  return 0;
}

__attribute__((noinline))
static void waitAndChangeOwnerForRead(const int hash, const int owner) {
  if (slowOwnershipCheck(owner))
    return;

  DMP_waitForSerialMode();

  // Steal ownerhip.
  int newowner = DMP_MOT_OWNER_SHARED;
#ifdef DMP_ENABLE_DATA_GROUPING
  if (DMPMAP->innerResource != NULL) {
    newowner = DMPmotOwnerFromResource(DMPMAP->innerResource);
  }
#endif

  DMPmot[hash] = newowner;
}

__attribute__((noinline))
static void waitAndChangeOwnerForWrite(const int hash, const int owner) {
  if (slowOwnershipCheck(owner))
    return;

  DMP_waitForSerialMode();

#ifndef DMP_DISABLE_SINGLE_THREADED_ALWAYS_SHARE
  if(DMPnumLiveThreads == 1) {
    DMPmot[hash] = DMP_MOT_OWNER_SHARED;
    return;
  }
#endif

  // Steal ownerhip.
  int newowner = DMPmotOwnerFromThread(DMPMAP);
#ifdef DMP_ENABLE_DATA_GROUPING
  if (DMPMAP->innerResource != NULL) {
    newowner = DMPmotOwnerFromResource(DMPMAP->innerResource);
  }
#endif

  DMPmot[hash] = newowner;
}

//-----------------------------------------------------------------------
// Loads/Stores
//-----------------------------------------------------------------------

struct LoadStoreTraits {
  static inline void doLoad(const int hash, const int owner) {
    if (unlikely(owner != DMP_MOT_OWNER_SHARED &&
                 owner != DMPmotOwnerFromThread(DMPMAP))) {
      waitAndChangeOwnerForRead(hash, owner);
    }
  }

  static inline void doStore(const int hash, const int owner) {
    if (unlikely(owner != DMPmotOwnerFromThread(DMPMAP))) {
      waitAndChangeOwnerForWrite(hash, owner);
    }
  }
};

//
// Contained Accesses
//

void DMPloadContained(void* addr) {
  const int hash  = DMPmotHash(addr);
  const int owner = DMPmot[hash];
  LoadStoreTraits::doLoad(hash, owner);
}

void DMPstoreContained(void* addr) {
  const int hash  = DMPmotHash(addr);
  const int owner = DMPmot[hash];
  LoadStoreTraits::doStore(hash, owner);
}

//
// Uncontained Accesses
//

void DMPloadRange(void* addr, size_t size) {
  MotIterator<LoadStoreTraits::doLoad>::foreach(addr, size);
}

void DMPstoreRange(void* addr, size_t size) {
  MotIterator<LoadStoreTraits::doStore>::foreach(addr, size);
}

//--------------------------------------------------------------
// LibC stubs
//--------------------------------------------------------------

void DMPmemset(void* addr, int val, size_t size) {
  DMPstoreRange(addr, size);
  memset(addr, val, size);
}

void DMPmemcpy(void* dst, const void* src, size_t size) {
  DMPloadRange((void*)src, size);
  DMPstoreRange(dst, size);
  memcpy(dst, src, size);
}

#endif  // DMP_ENABLE_MODEL_O_S
