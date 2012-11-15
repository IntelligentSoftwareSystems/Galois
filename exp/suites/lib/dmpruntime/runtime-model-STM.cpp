// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Runtime
//   STM :: RunOwnership -> WaitForSerial -> RunSerial
//
// This is an experiment to count how many rollbacks a
// naive STM implementation hacked onto DMP would have.
//

#include "config.h"

#ifdef DMP_ENABLE_MODEL_STM

#include "dmp-internal.h"
#include "dmp-internal-mot.h"

#include <ext/hash_set>
typedef __gnu_cxx::hash_set<int> HashSet;

//--------------------------------------------------------------
// Encoding of MOT[x]:
//   High 16 bits: encode writes
//   Low 16 bits:  encode readers
//--------------------------------------------------------------

static inline uint32_t DMPthreadToReader(DmpThreadInfo* dmp) {
  return (dmp->stmThreadId >= 16) ? 0 : 1 << dmp->stmThreadId;
}

static inline uint32_t DMPthreadToWriter(DmpThreadInfo* dmp) {
  return (dmp->stmThreadId >= 16) ? 0 : 1 << (16 + dmp->stmThreadId);
}

static void DMPupdateStmThreadIds() {
  DmpThreadInfo* dmp = DMPfirstRunnable;
  if (dmp == NULL)
    return;
  for (int id = 0; ; ++id) {
    dmp->stmThreadId = id;
    dmp = dmp->nextRunnable;
    if (dmp == DMPfirstRunnable)
      break;
  }
}

//-----------------------------------------------------------------------
// API
//-----------------------------------------------------------------------

uint64_t DMPstmParallelQuanta;
uint64_t DMPstmSerialQuanta;

static HashSet ToClear;

__thread HashSet* DMPreadLog;
__thread HashSet* DMPwriteLog;

void DMP_initRuntime() {
  memset(DMPmot, 0, sizeof DMPmot);
}

void DMP_initRuntimeThread() {
  DMPreadLog  = new HashSet;
  DMPwriteLog = new HashSet;
  DMPMAP->stmReadLog  = (void*)DMPreadLog;
  DMPMAP->stmWriteLog = (void*)DMPwriteLog;
}

extern void __DMPthread_addToRunnableQueue__(DmpThreadInfo* dmp);
extern void __DMPthread_removeFromRunnableQueue__(DmpThreadInfo* dmp);

void DMPthread_addToRunnableQueue(DmpThreadInfo* dmp) {
  __DMPthread_addToRunnableQueue__(dmp);
  DMPupdateStmThreadIds();
}

void DMPthread_removeFromRunnableQueue(DmpThreadInfo* dmp) {
  __DMPthread_removeFromRunnableQueue__(dmp);
  DMPupdateStmThreadIds();
}

void DMPstmCheckConflictsForThread(DmpThreadInfo* dmp) {
  HashSet* readLog  = (HashSet*)dmp->stmReadLog;
  HashSet* writeLog = (HashSet*)dmp->stmWriteLog;

  dmp->stmConflicts = 0;
  if (DMPnumRunnableThreads < 2)
    return;

  const uint32_t readconflict  = 0xffff & ~DMPthreadToReader(dmp);
  const uint32_t writeconflict = (0xffff << 16) & ~DMPthreadToWriter(dmp);
  bool conflict = false;

  // Check for R/W conflicts with other threads.
  for (HashSet::iterator hash = readLog->begin(); hash != readLog->end(); ++hash) {
    const uint32_t rc = (DMPmot[*hash] & readconflict);
    const uint32_t wc = (DMPmot[*hash] & writeconflict);
    if (wc != 0) {
      dmp->stmConflicts |= (wc >> 16);
    }
    DMPmot[*hash] |= DMPthreadToReader(dmp);
    ToClear.insert(*hash);
  }

  // Check for R/W and W/W conflicts with other threads.
  for (HashSet::iterator hash = writeLog->begin(); hash != writeLog->end(); ++hash) {
    const uint32_t rc = (DMPmot[*hash] & readconflict);
    const uint32_t wc = (DMPmot[*hash] & writeconflict);
    if (rc != 0) {
      dmp->stmConflicts |= rc;
    }
    if (wc != 0) {
      dmp->stmConflicts |= (wc >> 16);
    }
    DMPmot[*hash] |= DMPthreadToWriter(dmp);
    ToClear.insert(*hash);
  }
}

void DMPstmCheckConflicts() {
  // Determine conflicts.
  DmpThreadInfo* dmp = DMPfirstRunnable;
  while (dmp != NULL) {
    DMPstmCheckConflictsForThread(dmp);
    dmp = dmp->nextRunnable;
    if (dmp == DMPfirstRunnable)
      break;
  }
  // Clear DMPmot[].
  for (HashSet::iterator h = ToClear.begin(); h != ToClear.end(); ++h)
    DMPmot[*h] = 0;
  ToClear.clear();
  // Clear logs.
  for (int i = 0; i < DMPthreadInfosSize; ++i) {
    ((HashSet*)DMPthreadInfos[i]->stmReadLog)->clear();
    ((HashSet*)DMPthreadInfos[i]->stmWriteLog)->clear();
  }
  // Now schedule the threads.
  if (DMPnumRunnableThreads >= 2) {
    bool didall = false;
    int scheduled = 0;
    int running = 0;
    for (int n = 0; !didall; ++n) {
      // Another scheduling iteration.
      if (n > 1) {
        DMPstmSerialQuanta++;
      } else if (n == 1) {
        DMPstmParallelQuanta++;
      }
      didall = true;
      running = 0;
      for (dmp = DMPfirstRunnable; dmp != NULL; ) {
        const int me = 1 << dmp->stmThreadId;
        // Scheduled yet?
        if ((me & scheduled) == 0) {
          didall = false;
          // Can I run with the current group?
          if ((running & dmp->stmConflicts) == 0) {
            scheduled |= me;
            running |= me;
          }
        }
        // Next.
        dmp = dmp->nextRunnable;
        if (dmp == DMPfirstRunnable)
          break;
      }
    }
  }
}

//-----------------------------------------------------------------------
// Loads/Stores
//-----------------------------------------------------------------------

struct LoadStoreTraits {
  static inline void doLoad(const int hash, const int dummy) {
    DMPreadLog->insert(hash);
  }

  static inline void doStore(const int hash, const int dummy) {
    DMPwriteLog->insert(hash);
  }
};

//
// Contained Accesses
//

void DMPloadContained(void* addr) {
  LoadStoreTraits::doLoad(DMPmotHash(addr), 0);
}

void DMPstoreContained(void* addr) {
  LoadStoreTraits::doStore(DMPmotHash(addr), 0);
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

#endif  // DMP_ENABLE_MODEL_STM
