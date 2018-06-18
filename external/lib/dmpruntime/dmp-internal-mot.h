// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Utils for ownership
// Always include after dmp-internal.h
//

#ifndef _DMP_INTERNAL_MOT_H_

//--------------------------------------------------------------
// The MOT
// The encoding of MOT[x] varies per execution model, but for
// parallel commit, we expect to own the encoding space within
// [0, MaxThreads].
//--------------------------------------------------------------

#ifndef DMP_MOT_GRANULARITY
#define DMP_MOT_GRANULARITY 6
#endif

#ifndef DMP_MOT_BITS
#if defined(__LP_64__) || defined(__LP64__) || defined(__amd64__) ||           \
    defined(__x64_64__)
#define DMP_MOT_BITS 28
#else
#define DMP_MOT_BITS (32 - DMP_MOT_GRANULARITY)
#endif
#endif

#define DMP_MOT_ENTRY_SIZE (1 << DMP_MOT_GRANULARITY)
#define DMP_MOT_ENTRIES (1 << DMP_MOT_BITS)
#define DMP_MOT_MASK (DMP_MOT_ENTRIES - 1)

extern uint32_t DMPmot[DMP_MOT_ENTRIES];

static inline int DMPmotHash(void* addr) {
  return ((((uintptr_t)addr) >> DMP_MOT_GRANULARITY) & DMP_MOT_MASK);
}

static inline void* DMPmotAddrFromHash(int hash) {
  return (void*)(((uintptr_t)hash) << DMP_MOT_GRANULARITY);
}

//-----------------------------------------------------------------------
// Iterator
//-----------------------------------------------------------------------

template <void Visit(const int hash, const int owner)>
struct MotIterator {
  //
  // Visit all MOT entries in the range [startAddr, startAddr+size).
  //

  static FORCE_INLINE void foreach (void* startAddr, size_t size) {
    const size_t sz = ((size) > 0) ? ((size)-1) : 0;
    int index       = (((uintptr_t)(startAddr)) >> DMP_MOT_GRANULARITY);
    int end         = (((uintptr_t)(startAddr) + sz) >> DMP_MOT_GRANULARITY);
    for (; index <= end; ++index) {
      const int hash  = index & DMP_MOT_MASK;
      const int owner = DMPmot[hash];
      Visit(hash, owner);
    }
  }
};

#endif // _DMP_INTERNAL_MOT_H_
