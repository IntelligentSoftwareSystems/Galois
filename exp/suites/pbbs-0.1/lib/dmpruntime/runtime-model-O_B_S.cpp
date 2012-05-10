// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Runtime
//   O|B|S :: RunOwnership
//             -> WaitForBuffered -> RunBuffered -> WaitForCommit -> RunCommit
//             -> WaitForSerial -> RunSerial -> WaitForOwnership
//

#include "config.h"

#ifdef DMP_ENABLE_MODEL_O_B_S

#include "dmp-internal.h"
#include "dmp-internal-mot.h"
#include "dmp-internal-wb.h"

#if defined(DMP_ENABLE_WB_READLOG) && defined(DMP_ENABLE_WB_PARALLEL_COMMIT)
#error "READLOG and PARALLEL_COMMIT are not compatible"
#endif

//--------------------------------------------------------------
// Encoding of MOT[x]
//
// High bits
//   -- Shared :: DMP_MOT_OWNER_SHARED
//   -- Normal :: the thread ID of the owner
//
// Low bits
//   These are used along with parallel commit for all MOT
//   entries currently "shared", to count how many threads
//   have touched the corresponding cacheline.
//--------------------------------------------------------------

#define DMP_MOT_OWNER_SHARED  (1 << 31)

static inline uint32_t DMPmotOwnerFromThread(DmpThreadInfo* dmp) {
  return dmp->threadID;
}

//
// Read Log
// Each entry is an MOT hash
//

#ifdef DMP_ENABLE_WB_READLOG
__thread BumpAllocator<int> DMPreadLog;
#endif

//-----------------------------------------------------------------------
// API
//-----------------------------------------------------------------------

void DMP_initRuntime() {
  dmp_static_assert(DMP_MOT_GRANULARITY == DMP_WB_GRANULARITY);
  dmp_static_assert(DMP_MOT_ENTRY_SIZE  == DMP_WB_ENTRY_SIZE);
  dmp_static_assert(MaxThreads < (uint32_t)DMP_MOT_OWNER_SHARED);
  // LLVM runs out of memory if we use a static initializer for this !!! :-(
  for (int i = 0; i < DMP_MOT_ENTRIES; ++i)
    DMPmot[i] = DMP_MOT_OWNER_SHARED;
}

void DMP_initRuntimeThread() {
  DMPwbInit();
#ifdef DMP_ENABLE_WB_READLOG
  DMPreadLog.init(1024);
#endif
}

void DMP_commitBufferedWrites() {
  //
  // Bring all nonlocal reads to 'shared'.
  //
#ifdef DMP_ENABLE_WB_READLOG
#ifndef DMP_DISABLE_SINGLE_THREADED_ALWAYS_SHARE
  if(likely(DMPnumLiveThreads > 1)) {
    for (BumpAllocator<int>::iterator
         h = DMPreadLog.getiterator(); h.isValid(); h.next()) {
      const int me = DMPmotOwnerFromThread(DMPMAP);
      const int hash = *h.get();
      const int owner = DMPmot[hash];
      if (owner != me && owner != DMP_MOT_OWNER_SHARED) {
        DMPmot[hash] = DMP_MOT_OWNER_SHARED;
      }
    }
  }
#endif
  DMPreadLog.clear();
#endif

  //
  // Commit all writes
  // Also, take ownership of all writes.
  //
  for (BumpAllocator<LogEntry>::iterator
       e = DMPwbPool.getiterator(); e.isValid(); e.next()) {
    // Start the commit.
    LogEntryCommitLock* lock = NULL;
    bool manyWriters = false;
    if (DMPwbCommitLogEntryStart(e.get(), &lock, &manyWriters)) {
      // The current MOT[x] should be shared.  Take ownership
      // of that location iff we have been its only writer.
#ifndef DMP_DISABLE_SINGLE_THREADED_ALWAYS_SHARE
      if(likely(DMPnumLiveThreads > 1)) {
        const int me = DMPmotOwnerFromThread(DMPMAP);
        const int hash = DMPmotHash(e.get()->base);
        if (manyWriters == false)
          DMPmot[hash] = me;
      }
#endif
    }
    // Finish the commit.
    DMPwbCommitLogEntryEnd(lock);
  }

  DMPwbCommitFlush();
}

//-----------------------------------------------------------------------
// Loads/Stores
//-----------------------------------------------------------------------

struct LoadStoreTraits {
  static inline void doLoad(const WbIterator* iter) {
    if (likely(DMPMAP->state == RunOwnership)) {
      // If we own it, we can read it directly.
      // Otherwise, we must block.
      const int me = DMPmotOwnerFromThread(DMPMAP);
      const int hash  = DMPmotHash(iter->base);
      const int owner = DMPmot[hash];
      if (likely(owner == me || (owner & DMP_MOT_OWNER_SHARED) != 0)) {
        memcpy(iter->buffer, iter->addr, iter->currSize);
        return;
      }
      DMP_waitForBufferingMode();
    }
    // In buffering mode, everything gets buffered, and we log reads.
    DMPwbDoLoad(iter);
#ifdef DMP_ENABLE_WB_READLOG
    *(DMPreadLog.alloc()) = DMPmotHash(base);
#endif
  }

  static inline void doStore(const WbIterator* iter) {
    if (likely(DMPMAP->state == RunOwnership)) {
      // If we own it, we can write it directly.
      // Otherwise, we must block.
      const int me = DMPmotOwnerFromThread(DMPMAP);
      const int hash  = DMPmotHash(iter->base);
      const int owner = DMPmot[hash];
      if (likely(owner == me)) {
        memcpy(iter->addr, iter->buffer, iter->currSize);
        return;
      }
      DMP_waitForBufferingMode();
    }
    // In buffering mode, everything gets buffered.
    DMPwbDoStore(iter);
  }

  static inline void doMemset(const WbIterator* iter, int val) {
    if (likely(DMPMAP->state == RunOwnership)) {
      // If we own it, we can write it directly.
      // Otherwise, we must block.
      const int me = DMPmotOwnerFromThread(DMPMAP);
      const int hash  = DMPmotHash(iter->base);
      const int owner = DMPmot[hash];
      if (likely(owner == me)) {
        memset(iter->addr, val, iter->currSize);
        return;
      }
      DMP_waitForBufferingMode();
    }
    // In buffering mode, everything gets buffered.
    DMPwbDoMemset(iter, val);
  }

  static inline void doMemcpy(const WbIterator* dst, const WbIterator* src) {
    if (likely(DMPMAP->state == RunOwnership)) {
      // If we have ownership, we can copy directly.
      // Otherwise, we must block.
      const int me = DMPmotOwnerFromThread(DMPMAP);
      const int dsthash = DMPmotHash(dst->base);
      const int srchash = DMPmotHash(src->base);
      const int dstowner = DMPmot[dsthash];
      const int srcowner = DMPmot[srchash];
      if (likely((srcowner == me || (srcowner & DMP_MOT_OWNER_SHARED) != 0)) &&
                 (dstowner == me)) {
        memcpy(dst->addr, src->addr, dst->currSize);
        return;
      }
      DMP_waitForBufferingMode();
    }
    // In buffering mode, everything gets buffered.
    DMPwbDoMemcpy(dst, src);
  }

  static void doChangeOwnerForLoad(const int hash, const int owner) {
    const int me = DMPmotOwnerFromThread(DMPMAP);
    if (unlikely(owner != me && owner != DMP_MOT_OWNER_SHARED))
      DMPmot[hash] = DMP_MOT_OWNER_SHARED;
  }

  static void doChangeOwnerForStore(const int hash, const int owner) {
    const int me = DMPmotOwnerFromThread(DMPMAP);
    if (unlikely(owner != me)) {
#ifndef DMP_DISABLE_SINGLE_THREADED_ALWAYS_SHARE
      if(unlikely(DMPnumLiveThreads == 1)) {
        DMPmot[hash] = DMP_MOT_OWNER_SHARED;
      } else
#endif
      DMPmot[hash] = me;
    }
  }
};

__attribute__((noinline))
static void doSerialLoad(void* addr, size_t size, void* outbuffer) {
  MotIterator<LoadStoreTraits::doChangeOwnerForLoad>::foreach(addr, size);
  memcpy(outbuffer, addr, size);
}

__attribute__((noinline))
static void doSerialStore(void* addr, size_t size, void* inbuffer) {
  MotIterator<LoadStoreTraits::doChangeOwnerForStore>::foreach(addr, size);
  memcpy(addr, inbuffer, size);
}

//
// Contained accesses
//

void DMPloadBufferedContained(void* addr, size_t size, void* outbuffer) {
  if (likely(DMPMAP->state == RunOwnership || DMPMAP->state == RunBuffered)) {
    WbIterator::doone<LoadStoreTraits::doLoad>(addr, size, outbuffer);
  } else {
    doSerialLoad(addr, size, outbuffer);
  }
}

void DMPstoreBufferedContained(void* addr, size_t size, void* inbuffer) {
  if (likely(DMPMAP->state == RunOwnership || DMPMAP->state == RunBuffered)) {
    WbIterator::doone<LoadStoreTraits::doStore>(addr, size, inbuffer);
  } else {
    doSerialStore(addr, size, inbuffer);
  }
}

//
// Uncontained accesses
//

void DMPloadBufferedRange(void* addr, size_t size, void* outbuffer) {
  if (likely(DMPMAP->state == RunOwnership || DMPMAP->state == RunBuffered)) {
    WbIterator::foreach<LoadStoreTraits::doLoad>(addr, size, outbuffer);
  } else {
    doSerialLoad(addr, size, outbuffer);
  }
}

void DMPstoreBufferedRange(void* addr, size_t size, void* inbuffer) {
  if (likely(DMPMAP->state == RunOwnership || DMPMAP->state == RunBuffered)) {
    WbIterator::foreach<LoadStoreTraits::doStore>(addr, size, inbuffer);
  } else {
    doSerialStore(addr, size, inbuffer);
  }
}

//
// Stack-allocation removal
//

void DMPremoveBufferedRange(void* addr, size_t size) {
  DMPwbRemoveRange(addr, size);
}

//
// Type specializations
//

template<typename T, bool isContained>
static inline T doLoadTyped(T* addr) {
  // Uncontained?
  if (!isContained && !DMPwbMemoryIsContained((void*)addr, sizeof(T))) {
    T tmp;
    DMPloadBufferedRange((void*)addr, sizeof(T), &tmp);
    return tmp;
  }
  // In ownership mode, if we own it, we can read it directly.
  // Otherwise, we must block.
  if (likely(DMPMAP->state == RunOwnership)) {
    const int me = DMPmotOwnerFromThread(DMPMAP);
    const int hash  = DMPmotHash(addr);
    const int owner = DMPmot[hash];
    if (likely(owner == me || owner == DMP_MOT_OWNER_SHARED)) {
      return *addr;
    }
    DMP_waitForBufferingMode();
  }
  // In buffering mode, everything gets buffered, and we log reads.
  if (likely(DMPMAP->state == RunBuffered)) {
#ifdef DMP_ENABLE_WB_READLOG
    *(DMPreadLog.alloc()) = DMPmotHash(addr);
#endif
    return DMPwbLoadTyped<T,isContained>(addr);
  }
  // Serial mode.
  const int hash  = DMPmotHash(addr);
  const int owner = DMPmot[hash];
  LoadStoreTraits::doChangeOwnerForLoad(hash, owner);
  return *addr;
}

template<typename T, bool isContained>
static inline void doStoreTyped(T* addr, T value) {
  // Uncontained?
  if (!isContained && !DMPwbMemoryIsContained((void*)addr, sizeof(T))) {
    T tmp = value;
    DMPstoreBufferedRange((void*)addr, sizeof(T), &tmp);
    return;
  }
  // In ownership mode, if we own it, we can write it directly.
  // Otherwise, we must block.
  if (likely(DMPMAP->state == RunOwnership)) {
    const int me = DMPmotOwnerFromThread(DMPMAP);
    const int hash  = DMPmotHash(addr);
    const int owner = DMPmot[hash];
    if (likely(owner == me)) {
      *addr = value;
      return;
    }
    DMP_waitForBufferingMode();
  }
  // In buffering mode, everything gets buffered.
  if (likely(DMPMAP->state == RunBuffered)) {
    DMPwbStoreTyped<T,isContained>(addr, value);
    return;
  }
  // Serial mode.
  const int hash  = DMPmotHash(addr);
  const int owner = DMPmot[hash];
  LoadStoreTraits::doChangeOwnerForStore(hash, owner);
  *addr = value;
}

#define INSTANTIATE(T, TNAME, KIND, CONTAINED)\
  T DMPloadBuffered ## KIND ## TNAME(T* addr) {\
    return doLoadTyped<T,CONTAINED>(addr);\
  }\
  void DMPstoreBuffered ## KIND ## TNAME(T* addr, T value) {\
    doStoreTyped<T,CONTAINED>(addr, value);\
  }

INSTANTIATE(uint8_t,  Int8,   Contained, true)
INSTANTIATE(uint16_t, Int16,  Contained, true)
INSTANTIATE(uint32_t, Int32,  Contained, true)
INSTANTIATE(uint64_t, Int64,  Contained, true)
INSTANTIATE(float,    Float,  Contained, true)
INSTANTIATE(double,   Double, Contained, true)
INSTANTIATE(void*,    Ptr,    Contained, true)

INSTANTIATE(uint16_t, Int16,  Range, false)
INSTANTIATE(uint32_t, Int32,  Range, false)
INSTANTIATE(uint64_t, Int64,  Range, false)
INSTANTIATE(float,    Float,  Range, false)
INSTANTIATE(double,   Double, Range, false)
INSTANTIATE(void*,    Ptr,    Range, false)

#undef INSTANTIATE

//--------------------------------------------------------------
// LibC stubs
//--------------------------------------------------------------

void DMPmemset(void* addr, int val, size_t size) {
  if (likely(DMPMAP->state == RunOwnership || DMPMAP->state == RunBuffered)) {
    WbIterator iter(addr, size);
    for (;;) {
      LoadStoreTraits::doMemset(&iter, val);
      if (!iter.next()) break;
    }
  } else {
    memset(addr, val, size);
  }
}

void DMPmemcpy(void* dst, const void* src, size_t size) {
  if (likely(DMPMAP->state == RunBuffered)) {
    WbIterator di(dst, size);
    WbIterator si((void*)src, size);
    WbIterator::initPair(&di, &si);
    for (;;) {
      LoadStoreTraits::doMemcpy(&di, &si);
      if (!WbIterator::nextPair(&di, &si)) break;
    }
  } else {
    memcpy(dst, src, size);
  }
}

#endif  // DMP_ENABLE_MODEL_O_B_S
