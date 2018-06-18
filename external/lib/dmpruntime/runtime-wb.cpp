// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Utils for write buffering
//

#include <algorithm>
#include "dmp-internal.h"

#ifdef DMP_ENABLE_BUFFERED_MODE
#include "dmp-internal-wb.h"
#include "dmp-internal-mot.h"

thread_local LogEntry* DMPwb[DMP_WB_HASHSIZE];  // the write buffer
thread_local BumpAllocator<LogEntry> DMPwbPool; // allocation pool
thread_local BumpAllocator<void*> DMPwbFreeLog; // log for calls to free()

//-----------------------------------------------------------------------
// Initialization
//-----------------------------------------------------------------------

static inline void DMPwbUpdateMOT(void* addr);
static inline void DMPwbFlushCommitLocks();

void DMPwbInit() {
  DMPwbResetQuantum();
  DMPwbPool.init(4096 * 8 / sizeof(LogEntry));
  DMPwbFreeLog.init(64);
  memset(DMPwb, 0, sizeof DMPwb);
}

void DMPwbResetQuantum() {
  if (DMPMAP->isLastRunnable)
    DMPwbFlushCommitLocks();
}

//-----------------------------------------------------------------------
// The Write Buffer Hash table
//-----------------------------------------------------------------------

#ifdef DMP_ENABLE_WB_MOVE_TO_FRONT

LogEntry* DMPwbLookup(void* base) {
  // Lookup the LogEntry of a base address, and move-to-front if found
  const int hash  = DMPwbHash(base);
  LogEntry* first = DMPwb[hash];
  for (LogEntry *p = NULL, *e = first; e != NULL; p = e, e = e->next)
    if (e->base == base) {
      if (p != NULL) {
        p->next     = e->next;
        e->next     = first;
        DMPwb[hash] = e;
      }
      return e;
    }
  return NULL;
}

#else // !DMP_ENABLE_WB_MOVE_TO_FRONT

LogEntry* DMPwbLookup(void* base) {
  // Lookup the LogEntry of a base address.
  const int hash = DMPwbHash(base);
  for (LogEntry* e = DMPwb[hash]; e != NULL; e = e->next)
    if (e->base == base)
      return e;
  return NULL;
}

#endif // DMP_ENABLE_WB_MOVE_TO_FRONT

LogEntry* DMPwbLookupOrAdd(void* base) {
  // Lookup the LogEntry of a base address, and add a new LogEntry if needed.
  LogEntry* e = DMPwbLookup(base);
  if (e != NULL)
    return e;
  const int hash = DMPwbHash(base);
  e              = DMPwbPool.alloc();
  e->base        = base;
  e->next        = DMPwb[hash];
  e->bitfield    = 0;
  DMPwb[hash]    = e;
  DMPwbUpdateMOT(base);
  // Prefetch the data[] from shared memory.
  memcpy(e->data, base, sizeof e->data);
  return e;
}

//-----------------------------------------------------------------------
// Bitfields
//-----------------------------------------------------------------------

#if (DMP_WB_GRANULARITY == 6)
#define MaxBitfield 0xffffffffffffffffllu
#else
#define MaxBitfield ((1llu << (1 << DMP_WB_GRANULARITY)) - 1)
#endif

static inline uint64_t DMPwbBitmaskFromSize(const int size) {
  if (size == 64)
    return 0xffffffffffffffffllu;
  else
    return ((1ull << size) - 1);
}

static inline int countbits(uint64_t b) {
  int c = 0;
  for (; b; c++)
    b &= b - 1;
  return c;
}

//-----------------------------------------------------------------------
// Commit
//-----------------------------------------------------------------------

static inline void DMPwbCopy8(char* to, char* from, int bitfield) {
  // REQUIRES: bitfield <= 0xff
  switch (bitfield) {
  case 0:
    // Fast path: write nothing.
    return;
  case 0xff:
    // Fast path: write everything.
    memcpy(to, from, 8);
    return;
  case 0x0f:
    // Fast path: write first 4 bytes.
    memcpy(to, from, 4);
    return;
  case 0xf0:
    // Fast path: write second 4 bytes.
    memcpy(to + 4, from + 4, 4);
    return;
  default:
    // Slow path: write each byte.
    for (; bitfield; to++, from++, bitfield >>= 1) {
      if (bitfield & 0x1)
        *to = *from;
    }
  }
}

static inline void DMPwbCopyN(char* to, char* from, uint64_t bitfield) {
  // Batch every 8 bytes.
  for (; bitfield; to += 8, from += 8, bitfield >>= 8) {
    DMPwbCopy8(to, from, (int)(bitfield & 0xff));
  }
}

static inline void DMPwbCopyLogEntry(char* to, char* from, uint64_t bitfield) {
#if (DMP_WB_GRANULARITY <= 3)
  DMPwbCopy8(to, from, bitfield);
#else
  DMPwbCopyN(to, from, bitfield);
#endif
}

#ifndef DMP_ENABLE_WB_PARALLEL_COMMIT

struct LogEntryCommitLock {};

bool DMPwbCommitLogEntryStart(LogEntry* e, LogEntryCommitLock** plock,
                              bool* manyWriters) {
  // Fast path: write nothing.
  if (e->bitfield == 0)
    return true;
  // Fast path: write everything.
  if (e->bitfield == MaxBitfield) {
    memcpy(e->base, e->data, sizeof e->data);
    return true;
  }
  // Slow path: batch every 8 bytes.
  DMPwbCopyLogEntry((char*)e->base, e->data, e->bitfield);
  return true;
}

void DMPwbCommitLogEntryEnd(LogEntryCommitLock* lock) {}

#endif

void DMPwbCommitFlush() {
  DMPwbPool.clear();
  memset(DMPwb, 0, sizeof DMPwb);
}

void DMPwbCommit() {
  for (BumpAllocator<LogEntry>::iterator e = DMPwbPool.getiterator();
       e.isValid(); e.next()) {
    LogEntryCommitLock* lock = NULL;
    DMPwbCommitLogEntryStart(e.get(), &lock, NULL);
    DMPwbCommitLogEntryEnd(lock);
  }
  DMPwbCommitFlush();
}

void DMPwbUpdateStats() {
  // Stats (ignoring the single-threaded case).
#ifdef DMP_ENABLE_INSTRUMENT_WORK
  if (DMPnumRunnableThreads == 1)
    return;
  // Count log size.
  if (DMPwbPool.size() > DMPMAP->wb_maxsize) {
    DMPMAP->wb_maxsize = DMPwbPool.size();
  }
  DMPMAP->wb_totalsize += DMPwbPool.size();
  DMPMAP->wb_totalquanta++;
  // Count used bytes.
  uint64_t used = 0;
  for (BumpAllocator<LogEntry>::iterator e = DMPwbPool.getiterator();
       e.isValid(); e.next()) {
    used += countbits(e.get()->bitfield);
  }
  DMPMAP->wb_totalused += used;
  if (used > DMPMAP->wb_maxused) {
    DMPMAP->wb_maxused = used;
  }
  // Count hash-chain sizes.
  uint64_t maxchain    = 0;
  uint64_t totalbucket = 0;
  for (int i = 0; i < DMP_WB_HASHSIZE; ++i) {
    int c = 0;
    for (LogEntry* e = DMPwb[i]; e != NULL; e = e->next) {
      c++;
    }
    if (c > 0)
      totalbucket++;
    if (c > maxchain)
      maxchain = c;
    DMPMAP->wb_totalhashchains += c;
  }
  DMPMAP->wb_totalhashbuckets += totalbucket;
  if (maxchain > DMPMAP->wb_maxhashchain) {
    DMPMAP->wb_maxhashchain = maxchain;
  }
#endif
}

//-----------------------------------------------------------------------
// Parallel Commit
//-----------------------------------------------------------------------

#ifdef DMP_ENABLE_WB_PARALLEL_COMMIT

//
// Commit locking
//

struct LogEntryCommitLock {
  atomic_int_t spinlock;
  LogEntry* first;
};

#define DMP_WB_COMMIT_LOCKS 1024

static LogEntryCommitLock DMPwbCommitLocks[DMP_WB_COMMIT_LOCKS];

static inline int DMPwbCommitLockHash(void* addr) {
  return (((uintptr_t)addr) >> DMP_WB_GRANULARITY) & (DMP_WB_COMMIT_LOCKS - 1);
}

static inline LogEntryCommitLock* DMPwbAcquireCommitLock(LogEntry* e) {
  LogEntryCommitLock* lock = &DMPwbCommitLocks[DMPwbCommitLockHash(e->base)];
  DMP_SPINLOCK_LOCK(&lock->spinlock);
  return lock;
}

static inline void DMPwbReleaseCommitLock(LogEntryCommitLock* lock) {
  DMP_SPINLOCK_UNLOCK(&lock->spinlock);
}

static inline void DMPwbFlushCommitLocks() {
  memset(DMPwbCommitLocks, 0, sizeof DMPwbCommitLocks);
}

//
// Using the MOT as a big bloom filter
//

#define DMP_WB_MOT_MASK (MaxThreads | (MaxThreads - 1))

static inline bool DMPwbCheckMOT(void* addr) {
  // Return true if there might be a conflict at 'addr'.
  return (DMPmot[DMPmotHash(addr)] & DMP_WB_MOT_MASK) > 1;
}

static inline void DMPwbUpdateMOT(void* addr) {
  // Happens concurrently.
  __sync_fetch_and_add(&DMPmot[DMPmotHash(addr)], 1);
}

static inline void DMPwbCleanupMOT(void* addr) {
  // We are private, so no sync ops needed.
#ifdef DMP_ENABLE_MODEL_B_S
  const int hash = DMPmotHash(addr);
  DMPmot[hash]   = 0;
#else
  const int hash = DMPmotHash(addr);
  const int hi   = DMPmot[hash] & ~DMP_WB_MOT_MASK;
  DMPmot[hash]   = hi;
#endif
}

static inline bool DMPwbDecrementMOT(void* addr) {
  // We are locked, so no sync ops needed.
  const int hash = DMPmotHash(addr);
  const int full = DMPmot[hash];
  const int hi   = full & ~DMP_WB_MOT_MASK;
  const int lo   = full & DMP_WB_MOT_MASK;
  if (lo == MaxThreads) {
    DMPmot[hash] = hi;
    return true;
  } else if (lo == 2) {
    DMPmot[hash] = hi | MaxThreads;
    return false;
  } else {
    DMPmot[hash] = hi | (lo - 1);
    return false;
  }
}

//
// Commit a single LogEntry
//

bool DMPwbCommitLogEntryStart(LogEntry* e, LogEntryCommitLock** plock,
                              bool* manyWriters) {
  // Fast path: write nothing.
  if (e->bitfield == 0)
    return false;
  // Check the bloom filters for a conflict.
  if (!DMPwbCheckMOT(e->base)) {
    DMPwbCleanupMOT(e->base);
    // Fast path: write everything.
    if (e->bitfield == MaxBitfield)
      memcpy(e->base, e->data, sizeof e->data);
    // Slow path: batch every 8 bytes.
    else
      DMPwbCopyLogEntry((char*)e->base, e->data, e->bitfield);
    return true;
  }
  // Stats.
#ifdef DMP_ENABLE_INSTRUMENT_WORK
  if (DMPnumLiveThreads > 1)
    DMPMAP->wb_totalcommitslocked++;
#endif
  // Lock this log entry.
  LogEntryCommitLock* lock = DMPwbAcquireCommitLock(e);
  // Let the caller know:
  // -- Are we the last writer (in nondeterministic real-time)?
  // -- Were there multiple writers?
  const bool islast = DMPwbDecrementMOT(e->base);
  // Search for previous conflicting writes and mask out all
  // writes from threads that occur at a logically-later time.
  // then publish this log entry.
  uint64_t bitfield = e->bitfield;
  for (LogEntry* n = lock->first; n != NULL && bitfield != 0; n = n->next) {
    if (n->base == e->base && n->threadID > DMPMAP->threadID)
      bitfield &= ~(n->bitfield);
    if (n->base == e->base && manyWriters != NULL)
      *manyWriters = true;
  }
  if (bitfield != 0) {
    // Fast path: write everything.
    if (bitfield == MaxBitfield)
      memcpy(e->base, e->data, sizeof e->data);
    // Slow path: batch every 8 bytes.
    else
      DMPwbCopyLogEntry((char*)e->base, e->data, bitfield);
    // Since we wrote something, link our log entry into the hash table.
    e->threadID = DMPMAP->threadID;
    e->next     = lock->first;
    lock->first = e;
  }
  *plock = lock;
  return islast;
}

void DMPwbCommitLogEntryEnd(LogEntryCommitLock* lock) {
  if (lock != NULL)
    DMPwbReleaseCommitLock(lock);
}

#else // !DMP_ENABLE_WB_PARALLEL_COMMIT

static inline void DMPwbFlushCommitLocks() {}
static inline void DMPwbUpdateMOT(void* addr) {}

#endif // DMP_ENABLE_WB_PARALLEL_COMMIT

#undef MaxBitfield

//-----------------------------------------------------------------------
// Loads/Stores
//-----------------------------------------------------------------------

FORCE_INLINE
void DMPwbDoLoad(const WbIterator* iter) {
  char* in          = iter->addr;
  char* out         = iter->buffer;
  const ptrdiff_t d = iter->currOffset;
  const ptrdiff_t s = iter->currSize;
  // Copy from 'addr' -> 'buffer'.
  LogEntry* e = DMPwbLookup(iter->base);
  if (e == NULL) {
    memcpy(out, in, s);
    return;
  }
  // Redirect 'addr' through the write buffer.
  char* inbuf = e->data + d;
  memcpy(out, inbuf, s);
}

FORCE_INLINE
void DMPwbDoStore(const WbIterator* iter) {
  // Copy from the side buffer -> the WB entry.
  const ptrdiff_t d = iter->currOffset;
  const ptrdiff_t s = iter->currSize;
  LogEntry* e       = DMPwbLookupOrAdd(iter->base);
  e->bitfield |= (DMPwbBitmaskFromSize(s) << d);
  memcpy((void*)(e->data + d), iter->buffer, s);
}

FORCE_INLINE
void DMPwbDoRemove(const WbIterator* iter) {
  // Make sure [addr, addr+s) has been removed from the write buffer.
  const ptrdiff_t d = iter->currOffset;
  const ptrdiff_t s = iter->currSize;
  LogEntry* e       = DMPwbLookup(iter->base);
  if (e != NULL)
    e->bitfield &= ~(DMPwbBitmaskFromSize(s) << d);
}

FORCE_INLINE
void DMPwbDoMemset(const WbIterator* iter, int val) {
  // Fill this write buffer entry.
  const ptrdiff_t d = iter->currOffset;
  const ptrdiff_t s = iter->currSize;
  LogEntry* e       = DMPwbLookupOrAdd(iter->base);
  e->bitfield |= (DMPwbBitmaskFromSize(s) << d);
  memset((void*)(e->data + d), val, s);
}

FORCE_INLINE
void DMPwbDoMemcpy(const WbIterator* dst, const WbIterator* src) {
  const ptrdiff_t s = dst->currSize;
  assert(dst->currSize == src->currSize);
  // Always store to the WB in 'dst'.
  LogEntry* e;
  e = DMPwbLookupOrAdd(dst->base);
  e->bitfield |= (DMPwbBitmaskFromSize(s) << dst->currOffset);
  char* out = e->data + dst->currOffset;
  // Load 'src' from main memory?
  e = DMPwbLookup(src->base);
  if (e == NULL) {
    memcpy(out, src->addr, s);
    return;
  }
  // Load 'src' from the WB.
  char* inbuf = e->data + src->currOffset;
  memcpy(out, inbuf, s);
}

//
// Contained accesses
//

void DMPwbLoadContained(void* addr, size_t size, void* outbuffer) {
  WbIterator::doone<DMPwbDoLoad>(addr, size, outbuffer);
}

void DMPwbStoreContained(void* addr, size_t size, void* inbuffer) {
  WbIterator::doone<DMPwbDoStore>(addr, size, inbuffer);
}

//
// Uncontained accesses
//

void DMPwbLoadRange(void* addr, size_t size, void* outbuffer) {
  WbIterator::foreach<DMPwbDoLoad>(addr, size, outbuffer);
}

void DMPwbStoreRange(void* addr, size_t size, void* inbuffer) {
  WbIterator::foreach<DMPwbDoStore>(addr, size, inbuffer);
}

//
// Dynamic-allocation removal
//

void DMPwbRemoveRange(void* addr, size_t size) {
  WbIterator::foreach<DMPwbDoRemove>(addr, size, NULL);
}

//
// Type specializations
// Uncontained typed accesses are redirected to an untyped ranged access.
//

// NOTE: LLVM occasionally lies about alignments of data larger than 4 bytes.
// For benchmarks where this happens, define the following macro as a
// workaround.
#ifdef DMP_ENABLE_WB_BAD_ALIGNMENTS
#define WBSIZECHECK (sizeof(T) > 4)
#else
#define WBSIZECHECK false
#endif

template <typename T, bool isContained>
inline T DMPwbLoadTyped(T* addr) {
  // Uncontained?
  if ((!isContained || WBSIZECHECK) &&
      !DMPwbMemoryIsContained((void*)addr, sizeof(T))) {
    T tmp;
    DMPwbLoadRange((void*)addr, sizeof(T), &tmp);
    return tmp;
  }
  // Get the log parameters.
  void* base  = DMPwbBase((void*)addr);
  ptrdiff_t d = (uintptr_t)addr - (uintptr_t)base;
  ptrdiff_t s = sizeof(T);
  // Check the write buffer.
  LogEntry* e = DMPwbLookup(base);
  if (e == NULL) {
    return *addr;
  }
  // Fetch from the write buffer.
  T* buf = (T*)(e->data + d);
  return *buf;
}

template <typename T, bool isContained>
inline void DMPwbStoreTyped(T* addr, T value) {
  // Uncontained?
  if ((!isContained || WBSIZECHECK) &&
      !DMPwbMemoryIsContained((void*)addr, sizeof(T))) {
    T tmp = value;
    DMPwbStoreRange((void*)addr, sizeof(T), &tmp);
    return;
  }
  // Get the log parameters.
  void* base  = DMPwbBase((void*)addr);
  ptrdiff_t d = (uintptr_t)addr - (uintptr_t)base;
  ptrdiff_t s = sizeof(T);
  // Copy 'value' into the write buffer.
  LogEntry* e = DMPwbLookupOrAdd(base);
  e->bitfield |= (DMPwbBitmaskFromSize(s) << d);
  T* buf = (T*)(e->data + d);
  *buf   = value;
}

// Instantiations.
#define INSTANTIATE(T)                                                         \
  template T DMPwbLoadTyped<T, true>(T*);                                      \
  template T DMPwbLoadTyped<T, false>(T*);                                     \
  template void DMPwbStoreTyped<T, true>(T*, T);                               \
  template void DMPwbStoreTyped<T, false>(T*, T);

INSTANTIATE(uint8_t)
INSTANTIATE(uint16_t)
INSTANTIATE(uint32_t)
INSTANTIATE(uint64_t)
INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(void*)

#undef INSTANTIATE

//--------------------------------------------------------------
// LibC stubs
//--------------------------------------------------------------

void DMPwbMemset(void* addr, int val, size_t size) {
  WbIterator iter(addr, size);
  for (;;) {
    DMPwbDoMemset(&iter, val);
    if (!iter.next())
      break;
  }
}

void DMPwbMemcpy(void* dst, const void* src, size_t size) {
  WbIterator di(dst, size);
  WbIterator si((void*)src, size);
  WbIterator::initPair(&di, &si);
  for (;;) {
    DMPwbDoMemcpy(&di, &si);
    if (!WbIterator::nextPair(&di, &si))
      break;
  }
}

#endif // DMP_ENABLE_BUFFERED_MODE
