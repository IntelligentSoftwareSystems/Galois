// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Runtime
//   OB|S :: RunBuffered -> WaitForCommit -> RunCommit
//            -> WaitForSerial -> RunSerial -> WaitForBuffered
//

#include "config.h"

#ifdef DMP_ENABLE_MODEL_OB_S

#include "dmp-internal.h"
#include "dmp-internal-mot.h"
#include "dmp-internal-wb.h"

//--------------------------------------------------------------
// Encoding of MOT[x]
//
// High bits
//   -- Always-shared :: DMP_MOT_OWNER_SHARED_ALWAYS
//   -- Shared        :: DMP_MOT_OWNER_SHARED
//   -- Normal        :: the thread ID of the owner
//
// Low bits
//   These are used along with parallel commit for all MOT
//   entries currently "shared", to count how many threads
//   have touched the corresponding cacheline.
//--------------------------------------------------------------

#define DMP_MOT_OWNER_SHARED_ALWAYS ((1 << 31))
#define DMP_MOT_OWNER_SHARED ((1 << 31) | (1 << 30))

static inline uint32_t DMPmotOwnerFromThread(DmpThreadInfo* dmp) {
  return dmp->threadID;
}

//-----------------------------------------------------------------------
// API
//-----------------------------------------------------------------------

void DMP_initRuntime() {
  dmp_static_assert(DMP_MOT_GRANULARITY == DMP_WB_GRANULARITY);
  dmp_static_assert(DMP_MOT_ENTRY_SIZE == DMP_WB_ENTRY_SIZE);
  dmp_static_assert(MaxThreads < (uint32_t)DMP_MOT_OWNER_SHARED);
  // LLVM runs out of memory if we use a static initializer for this !!! :-(
  for (int i = 0; i < DMP_MOT_ENTRIES; ++i)
    DMPmot[i] = DMP_MOT_OWNER_SHARED;
}

void DMP_initRuntimeThread() { DMPwbInit(); }

void DMP_commitBufferedWrites() {
  for (BumpAllocator<LogEntry>::iterator e = DMPwbPool.getiterator();
       e.isValid(); e.next()) {
    // Start the commit.
    LogEntryCommitLock* lock = NULL;
    bool manyWriters         = false;
    if (DMPwbCommitLogEntryStart(e.get(), &lock, &manyWriters)) {
      // The current MOT[x] should be shared.  Take ownership
      // of that location iff we have been its only writer.
#ifndef DMP_DISABLE_SINGLE_THREADED_ALWAYS_SHARE
      if (likely(DMPnumLiveThreads > 1)) {
        const int me    = DMPmotOwnerFromThread(DMPMAP);
        const int hash  = DMPmotHash(e.get()->base);
        const int owner = DMPmot[hash];
        if (owner == DMP_MOT_OWNER_SHARED) {
          if (manyWriters == false)
            DMPmot[hash] = me;
          else
            DMPmot[hash] = DMP_MOT_OWNER_SHARED_ALWAYS;
        }
      }
#endif
    }
    // Finish the commit.
    DMPwbCommitLogEntryEnd(lock);
  }
  DMPwbCommitFlush();
}

//-----------------------------------------------------------------------
// MOT Policy
// -- Each MOT[x] starts "shared"
// -- If written by a thread T0, MOT[x] becomes "owned-by-T0"
// -- If written or read by another thread T1, MOT[x] becomes "owned-by-T1"
//-----------------------------------------------------------------------

__attribute__((noinline)) static void waitAndChangeOwnerForRead(int hash,
                                                                int owner) {
  DMP_waitForSerialMode();
  DMPmot[hash] = DMP_MOT_OWNER_SHARED_ALWAYS;
}

__attribute__((noinline)) static void waitAndChangeOwnerForWrite(int hash,
                                                                 int owner) {
  DMP_waitForSerialMode();

#ifndef DMP_DISABLE_SINGLE_THREADED_ALWAYS_SHARE
  if (unlikely(DMPnumLiveThreads == 1)) {
    DMPmot[hash] = DMP_MOT_OWNER_SHARED;
  } else
#endif
      if (owner == DMP_MOT_OWNER_SHARED) {
    DMPmot[hash] = DMPmotOwnerFromThread(DMPMAP);
  } else {
    DMPmot[hash] = DMP_MOT_OWNER_SHARED_ALWAYS;
  }
}

//-----------------------------------------------------------------------
// Loads/Stores
//-----------------------------------------------------------------------

struct LoadStoreTraits {
  static inline void doLoad(const WbIterator* iter) {
    char* in          = iter->addr;
    char* out         = iter->buffer;
    const ptrdiff_t s = iter->currSize;
    // Copy from 'addr' -> 'buffer'.
    const int me    = DMPmotOwnerFromThread(DMPMAP);
    const int hash  = DMPmotHash(iter->base);
    const int owner = DMPmot[hash];
    if (likely(owner == me)) {
      // Local: read from memory
      memcpy(out, in, s);
    } else if (likely((owner & DMP_MOT_OWNER_SHARED) != 0 &&
                      DMPMAP->state == RunBuffered)) {
      // Shared: read from buffer if in buffering mode
      DMPwbDoLoad(iter);
    } else {
      // Nonlocal: block for serial mode
      waitAndChangeOwnerForRead(hash, owner);
      memcpy(out, in, s);
    }
  }

  static inline void doStore(const WbIterator* iter) {
    char* in          = iter->buffer;
    char* out         = iter->addr;
    const ptrdiff_t s = iter->currSize;
    // Copy from 'buffer' -> 'addr'.
    const int me    = DMPmotOwnerFromThread(DMPMAP);
    const int hash  = DMPmotHash(iter->base);
    const int owner = DMPmot[hash];
    if (likely(owner == me)) {
      // Local: write to memory
      memcpy(out, in, s);
    } else if (likely((owner & DMP_MOT_OWNER_SHARED) != 0 &&
                      DMPMAP->state == RunBuffered)) {
      // Shared: write to the buffer if in buffering mode.
      DMPwbDoStore(iter);
    } else {
      // Nonlocal: block for serial mode
      waitAndChangeOwnerForWrite(hash, owner);
      memcpy(out, in, s);
    }
  }

  static inline void doMemset(const WbIterator* iter, int val) {
    // Fill 'addr' with 'val'.
    const ptrdiff_t s = iter->currSize;
    const int me      = DMPmotOwnerFromThread(DMPMAP);
    const int hash    = DMPmotHash(iter->base);
    const int owner   = DMPmot[hash];
    if (likely(owner == me)) {
      // Local: write to memory
      memset(iter->addr, val, s);
    } else if (likely((owner & DMP_MOT_OWNER_SHARED) != 0 &&
                      DMPMAP->state == RunBuffered)) {
      // Shared: write to the buffer if in buffering mode.
      DMPwbDoMemset(iter, val);
    } else {
      // Nonlocal: block for serial mode
      waitAndChangeOwnerForWrite(hash, owner);
      memset(iter->addr, val, s);
    }
  }

  static inline void doMemcpy(WbIterator* dst, WbIterator* src) {
    // Fill 'addr' with 'val;.
    const ptrdiff_t s  = dst->currSize;
    const int me       = DMPmotOwnerFromThread(DMPMAP);
    const int dsthash  = DMPmotHash(dst->base);
    const int srchash  = DMPmotHash(src->base);
    const int dstowner = DMPmot[dsthash];
    const int srcowner = DMPmot[srchash];
    if (likely(dstowner == me && srcowner == me)) {
      // Local: copy to and from memory
      memcpy(dst->addr, src->addr, s);
    } else if (likely((dstowner & DMP_MOT_OWNER_SHARED) != 0 &&
                      (srcowner & DMP_MOT_OWNER_SHARED) != 0 &&
                      DMPMAP->state == RunBuffered)) {
      // Shared: copy to and from the buffer.
      DMPwbDoMemcpy(dst, src);
    } else if (likely(dstowner == me &&
                      (srcowner & DMP_MOT_OWNER_SHARED) != 0 &&
                      DMPMAP->state == RunBuffered)) {
      // Partially shared: copy from the buffer to memory.
      src->buffer = dst->addr;
      DMPwbDoLoad(src);
    } else if (likely((dstowner & DMP_MOT_OWNER_SHARED) != 0 &&
                      srcowner == me && DMPMAP->state == RunBuffered)) {
      // Partially shared: copy from memory to the buffer.
      dst->buffer = src->addr;
      DMPwbDoStore(dst);
    } else {
      // Nonlocal: block for serial mode
      waitAndChangeOwnerForRead(srchash, srcowner);
      waitAndChangeOwnerForWrite(dsthash, dstowner);
      memcpy(dst->addr, src->addr, s);
    }
  }
};

//
// Contained accesses
//

void DMPloadBufferedContained(void* addr, size_t size, void* outbuffer) {
  WbIterator::doone<LoadStoreTraits::doLoad>(addr, size, outbuffer);
}

void DMPstoreBufferedContained(void* addr, size_t size, void* inbuffer) {
  WbIterator::doone<LoadStoreTraits::doStore>(addr, size, inbuffer);
}

//
// Uncontained accesses
//

void DMPloadBufferedRange(void* addr, size_t size, void* outbuffer) {
  WbIterator::foreach<LoadStoreTraits::doLoad>(addr, size, outbuffer);
}

void DMPstoreBufferedRange(void* addr, size_t size, void* inbuffer) {
  WbIterator::foreach<LoadStoreTraits::doStore>(addr, size, inbuffer);
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

template <typename T, bool isContained>
static inline T doLoadTyped(T* addr) {
  // Uncontained?
  if (!isContained && !DMPwbMemoryIsContained((void*)addr, sizeof(T))) {
    T tmp;
    DMPloadBufferedRange((void*)addr, sizeof(T), &tmp);
    return tmp;
  }
  // Check the MOT.
  const int me    = DMPmotOwnerFromThread(DMPMAP);
  const int hash  = DMPmotHash(addr);
  const int owner = DMPmot[hash];
  if (likely(owner == me)) {
    // Local: read from memory
    return *addr;
  } else if (likely((owner & DMP_MOT_OWNER_SHARED) != 0)) {
    // Shared: read from buffer if not in serial mode.
    if (likely(DMPMAP->state == RunBuffered))
      return DMPwbLoadTyped<T, isContained>(addr);
    else
      return *addr;
  } else {
    // Nonlocal: block for serial mode
    waitAndChangeOwnerForRead(hash, owner);
    return *addr;
  }
}

template <typename T, bool isContained>
static inline void doStoreTyped(T* addr, T value) {
  // Uncontained?
  if (!isContained && !DMPwbMemoryIsContained((void*)addr, sizeof(T))) {
    T tmp = value;
    DMPstoreBufferedRange((void*)addr, sizeof(T), &tmp);
    return;
  }
  // Check the MOT.
  const int me    = DMPmotOwnerFromThread(DMPMAP);
  const int hash  = DMPmotHash(addr);
  const int owner = DMPmot[hash];
  if (likely(owner == me)) {
    // Local: write to memory
    *addr = value;
  } else if (likely((owner & DMP_MOT_OWNER_SHARED) != 0 &&
                    DMPMAP->state == RunBuffered)) {
    // Shared: write to the buffer if not in serial mode.
    DMPwbStoreTyped<T, isContained>(addr, value);
  } else {
    // Nonlocal: block for serial mode
    waitAndChangeOwnerForWrite(hash, owner);
    *addr = value;
  }
}

#define INSTANTIATE(T, TNAME, KIND, CONTAINED)                                 \
  T DMPloadBuffered##KIND##TNAME(T* addr) {                                    \
    return doLoadTyped<T, CONTAINED>(addr);                                    \
  }                                                                            \
  void DMPstoreBuffered##KIND##TNAME(T* addr, T value) {                       \
    doStoreTyped<T, CONTAINED>(addr, value);                                   \
  }

INSTANTIATE(uint8_t, Int8, Contained, true)
INSTANTIATE(uint16_t, Int16, Contained, true)
INSTANTIATE(uint32_t, Int32, Contained, true)
INSTANTIATE(uint64_t, Int64, Contained, true)
INSTANTIATE(float, Float, Contained, true)
INSTANTIATE(double, Double, Contained, true)
INSTANTIATE(void*, Ptr, Contained, true)

INSTANTIATE(uint16_t, Int16, Range, false)
INSTANTIATE(uint32_t, Int32, Range, false)
INSTANTIATE(uint64_t, Int64, Range, false)
INSTANTIATE(float, Float, Range, false)
INSTANTIATE(double, Double, Range, false)
INSTANTIATE(void*, Ptr, Range, false)

#undef INSTANTIATE

//--------------------------------------------------------------
// LibC stubs
//--------------------------------------------------------------

void DMPmemset(void* addr, int val, size_t size) {
  if (likely(DMPMAP->state == RunBuffered)) {
    WbIterator iter(addr, size);
    for (;;) {
      LoadStoreTraits::doMemset(&iter, val);
      if (!iter.next())
        break;
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
      if (!WbIterator::nextPair(&di, &si))
        break;
    }
  } else {
    memcpy(dst, src, size);
  }
}

#endif // DMP_ENABLE_MODEL_OB_S
