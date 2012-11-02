// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Utils for write buffering
// Always include after dmp-internal.h
//

#ifndef _DMP_INTERNAL_WB_H_

#include <algorithm>

//-----------------------------------------------------------------------
// Bump allocator
// This is thread-local, so it cannot have a constructor!
//-----------------------------------------------------------------------

template<typename T>
struct BumpAllocator {
  T* alloc() {
    if (unlikely(currItem == itemsPerPage)) {
      if (++currPage == numPages)
        addpage();
      currItem = 0;
    }
    T* item = pages[currPage] + currItem;
    currItem++;
    return item;
  }

  void init(int _itemsPerPage) {
    itemsPerPage = _itemsPerPage;
    currItem = 0;
    currPage = 0;
    numPages = 0;
    numSlots = 0;
    pages = NULL;
    addpage();
  }

  void clear() {
    currItem = 0;
    currPage = 0;
  }
  int size() const {
    return currPage*itemsPerPage + currItem;
  }

  struct iterator {
    iterator(const BumpAllocator* b)
      : bump(b), item(0), page(0)
    {}

    const BumpAllocator* bump;
    int item;
    int page;

    T* get() const {
      return bump->pages[page] + item;
    }

    void next() {
      if (++item >= bump->itemsPerPage) {
        item = 0;
        page++;
      }
    }

    bool isValid() const {
      return (page == bump->currPage && item < bump->currItem) ||
             (page < bump->currPage);
    }
  };

  iterator getiterator() const { return iterator(this); }

  // === Below Here Is Private ===
  // If we declare it private, gcc complains this class isn't a POD (ugh).

  void addpage() {
    // Make sure there's a slot in pages_[]
    if (numPages == numSlots) {
      if (numSlots == 0)
        numSlots = 4;
      else
        numSlots *= 2;
      pages = (T**)real_realloc(pages, numSlots * sizeof(pages[0]));
      memset(pages+numPages, 0, (numSlots - numPages) * sizeof(pages[0]));
    }
    // Alloc the new page, leaving it uninitalized.
    pages[numPages++] = (T*)real_malloc(itemsPerPage * sizeof(T));
  }

  // The next item is allocated to pages[currPage][currItem].
  int currItem;
  int currPage;
  int itemsPerPage;
  // The allocated pages (with a capacity for "numSlots" pages).
  int numPages;
  int numSlots;
  T** pages;
};

//-----------------------------------------------------------------------
// Write Buffer
//-----------------------------------------------------------------------

#ifndef DMP_WB_GRANULARITY
# ifdef DMP_MOT_GRANULARITY
#  define DMP_WB_GRANULARITY DMP_MOT_GRANULARITY
# else
#  define DMP_WB_GRANULARITY 6
# endif
#endif

#ifdef DMP_ENABLE_BUFFERED_MODE
#if (DMP_WB_GRANULARITY < 2) || (6 < DMP_WB_GRANULARITY)
#error "DMP_WB_GRANULARITY is out-of-range"
#endif
#endif

#ifndef DMP_WB_HASHSIZE
#define DMP_WB_HASHSIZE 1024    // number of hash buckets (power-of-2)
#endif

#define DMP_WB_HASHMASK    (DMP_WB_HASHSIZE - 1)
#define DMP_WB_ENTRY_SIZE  (1 << DMP_WB_GRANULARITY)

struct LogEntry {
  void*     base;       // base address
  LogEntry* next;       // linked list
  uint64_t  bitfield;   // bitfield of valid bytes in data[]
#ifndef DMP_ENABLE_WB_PARALLEL_COMMIT
  char      data[DMP_WB_ENTRY_SIZE];
#else
  union {
    char     data[DMP_WB_ENTRY_SIZE];  // normal usage
    uint32_t threadID;                 // used after the data has been published
  };
#endif
};

extern __thread LogEntry* DMPwb[DMP_WB_HASHSIZE];   // the write buffer
extern __thread BumpAllocator<LogEntry> DMPwbPool;  // allocation pool

//-----------------------------------------------------------------------
// Write Buffer API
//-----------------------------------------------------------------------

static inline void* DMPwbBase(void* addr) {
  return (void*)((uintptr_t)addr & ~(DMP_WB_ENTRY_SIZE - 1));
}

static inline int DMPwbHash(void* addr) {
  return ((((uintptr_t) addr) >> DMP_WB_GRANULARITY) & DMP_WB_HASHMASK);
}

static inline int DMPwbMemoryIsContained(void* addr, size_t size) {
  return ((uintptr_t)addr - (uintptr_t)DMPwbBase(addr) + size) <= DMP_WB_ENTRY_SIZE;
}

LogEntry* DMPwbLookup(void* base);
LogEntry* DMPwbLookupOrAdd(void* base);

void DMPwbInit();
void DMPwbCommit();
void DMPwbCommitFlush();
void DMPwbUpdateStats();
void DMPwbResetQuantum();

void DMPwbLogDeallocation(void* addr);
void DMPwbFlushDeallocations();

// Usage:
//   lock = NULL
//   manyWriters = false
//   if (DMPwbCommitLogEntryStart(e, &lock, &manyWriters)) {
//      // It is safe to update things related to 'e'.
//      // 'manyWriters == true' only if 2+ threads wrote to 'e' during this quantum.
//   }
//   DMPwbCommitLogEntryEnd(lock)
struct LogEntryCommitLock;
bool DMPwbCommitLogEntryStart(LogEntry *e, LogEntryCommitLock** plock, bool* manyWriters);
void DMPwbCommitLogEntryEnd(LogEntryCommitLock *lock);

void DMPwbLoadContained(void* addr, size_t size, void* outbuffer);
void DMPwbStoreContained(void* addr, size_t size, void* inbuffer);

void DMPwbLoadRange(void* addr, size_t size, void* outbuffer);
void DMPwbStoreRange(void* addr, size_t size, void* inbuffer);
void DMPwbRemoveRange(void* addr, size_t size);

template<typename T, bool isContained> T DMPwbLoadTyped(T* addr);
template<typename T, bool isContained> void DMPwbStoreTyped(T* addr, T value);

// LibC stubs
void DMPwbMemset(void* addr, int val, size_t size);
void DMPwbMemcpy(void* dst, const void* src, size_t size);

// internals
struct WbIterator;
void DMPwbDoLoad(const WbIterator* iter);
void DMPwbDoStore(const WbIterator* iter);
void DMPwbDoMemset(const WbIterator* iter, int val);
void DMPwbDoMemcpy(const WbIterator* dst, const WbIterator* src);

//-----------------------------------------------------------------------
// Iterator
//-----------------------------------------------------------------------

struct WbIterator {
  inline WbIterator(void* startAddr, size_t startSize, void* startBuffer = NULL)
    : buffer((char*)startBuffer), addr((char*)startAddr), remaining(startSize)
  { update(); }

  //
  // This iterates over a range of addresses in the WB,
  //   [startAddr, startAddr + startSize),
  // and optionally iterates over a corresponding range in a side buffer,
  //   [startBuffer, startBuffer + startSize).
  //

  char* buffer;      // current address inside the side buffer (if any)
  char* addr;        // current address inside the WB
  char* base;        // base address of the current WB entry
  ptrdiff_t remaining;   // remaining bytes in the range
  ptrdiff_t currOffset;  // offset in the current WB entry
  ptrdiff_t currSize;    // length in the current WB entry

  inline bool next() {
    if (currSize >= remaining)
      return false;
    remaining -= currSize;
    addr += currSize;
    buffer += currSize;
    update();
    return true;
  }

  inline void update() {
    base = (char*)DMPwbBase(addr);
    currOffset = (uintptr_t)addr - (uintptr_t)base;
    currSize = std::min(remaining, (ptrdiff_t)(DMP_WB_ENTRY_SIZE - currOffset));
  }

  //
  // Visit all WB entries in the range.
  //

  template <void Visit(const WbIterator* iter)>
  static FORCE_INLINE
  void foreach(void* startAddr, size_t startSize, void* startBuffer) {
    WbIterator iter(startAddr, startSize, startBuffer);
    for (;;) {
      Visit(&iter);
      if (!iter.next()) break;
    }
  }

  //
  // Visit a single WB entry.
  // This is like foreach(), but only visits the first entry.
  //

  template <void Visit(const WbIterator* iter)>
  static FORCE_INLINE
  void doone(void* startAddr, size_t startSize, void* startBuffer) {
    WbIterator iter(startAddr, startSize, startBuffer);
    Visit(&iter);
  }

  //
  // For iterating over two iterators in sync.
  //

  static inline void initPair(WbIterator* a, WbIterator* b) {
    // Call just after construction.
    if (a->currSize > b->currSize) {
      a->currSize = b->currSize;
    } else if (b->currSize > a->currSize) {
      b->currSize = a->currSize;
    }
  }

  static inline bool nextPair(WbIterator* a, WbIterator* b) {
    // Call at each iteration.
    if (a->currSize >= a->remaining) {
      assert(b->currSize >= b->remaining);
      return false;
    }
    if (a->currOffset >= b->currOffset) {
      a->remaining -= a->currSize;
      b->remaining -= a->currSize;
      a->addr += a->currSize;
      b->addr += a->currSize;
      a->buffer += a->currSize;
      b->buffer += a->currSize;
      a->update();
      b->update();
      if (b->currSize < a->currSize)
        a->currSize = b->currSize;
      else
        assert(a->currSize == b->currSize);
      return true;
    } else {
      return nextPair(b, a);
    }
  }
};

#endif  // _DMP_INTERNAL_WB_H_
