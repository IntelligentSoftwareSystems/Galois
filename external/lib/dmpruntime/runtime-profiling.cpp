// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Runtime - debugging and instrumentation for profiling stats
//

#include <math.h>
#include <time.h>
#include "dmp-internal.h"
#ifdef DMP_ENABLE_BUFFERED_MODE
#include "dmp-internal-wb.h"
#endif


#ifdef DMP_ENABLE_QUANTUM_TIMING
bool DMPQuantumEndedWithAcquire = false;
#endif

//--------------------------------------------------------------
// Instrumenting Resource Acquires
//--------------------------------------------------------------

#ifdef DMP_ENABLE_INSTRUMENT_ACQUIRES

typedef struct OwnershipRecord OwnershipRecord;
struct OwnershipRecord {
  int id;           // address hash or mutex ID
  int nhistory;     // ownership history
  int history[256];
};

#define HistorySize 4096
static OwnershipRecord dataHistories[HistorySize];
static OwnershipRecord resourceHistories[HistorySize];

static OwnershipRecord* find_record(OwnershipRecord* histories, const int id) {
  int i;
  for (i = 0; i < HistorySize; ++i) {
    if (histories[i].nhistory == 0)
      break;
    if (histories[i].id == id)
      break;
  }
  return i < HistorySize ? histories + i : NULL;
}

static void update_record(OwnershipRecord* r, const int id, const int owner) {
  if (r->nhistory < 256) {
    r->id = id;
    r->history[r->nhistory++] = owner;
  }
}

static void instrument_owner_change(int hash, int oldowner, int newowner) {
  /*if (DMPnumRunnableThreads >= DMP_NUM_PHYSICAL_PROCESSORS) {
    OwnershipRecord* r = find_record(dataHistories, hash);
    // Don't include the first SHARED -> PRIVATE transition.
    if (r != NULL && (r->nhistory > 0 || oldowner != DMP_MOT_OWNER_SHARED)) {
      if (r->nhistory == 0) update_record(r, hash, oldowner);
      update_record(r, hash, newowner);
    }
  }*/
}

static void instrument_resource_acquire(int id, int newowner) {
  if (DMPnumRunnableThreads >= DMP_NUM_PHYSICAL_PROCESSORS) {
    OwnershipRecord* r = find_record(resourceHistories, id);
    if (r != NULL) update_record(r, id, newowner);
  }
}

static uint64_t DMProundsWithSharedMutexes = 0;

void DMPinstrument_resource_acquire(DMPresource* r, int oldowner) {
  const int owner = DMPresource_owner(r);
  instrument_resource_acquire(DMPdataOwnerFromResource(r), owner);
  if (oldowner != owner && r->lastRoundAcquired == DMProundNumber) {
    DMProundsWithSharedMutexes++;
  }
  r->lastRoundAcquired = DMProundNumber;
}

#endif  // DMP_ENABLE_INSTRUMENT_ACQUIRES

//--------------------------------------------------------------
// Instrumenting Quantum Timing
//--------------------------------------------------------------

#if defined(DMP_ENABLE_QUANTUM_TIMING) || defined(DMP_ENABLE_ROUND_TIMING)

//#define DMP_QUANTUM_TIMING_CLOCK  CLOCK_MONOTONIC
#define DMP_QUANTUM_TIMING_CLOCK  CLOCK_THREAD_CPUTIME_ID

static inline uint64_t timespec_to_nanoseconds(struct timespec* ts) {
  return (uint64_t)ts->tv_sec * 1000000000ull + (uint64_t)ts->tv_nsec;
}

#endif

#ifdef DMP_ENABLE_QUANTUM_TIMING

static uint64_t DMPtimingRoundCount;

static void update_quantum_timings() {
  DMPtimingRoundCount++;

  //
  // For each thread, compute the percent time spent in each
  // state in this round.  Use this to update 'mean' and 'm2'
  // for each state.
  //

  for (int id = 0; id < DMPthreadInfosSize; ++id) {
    DmpThreadInfo* dmp = DMPthreadInfos[id];
    if (dmp->exited || dmp->state == Sleeping)
      continue;

    dmp->timing_total_quanta++;

    // Total.
    double total = 0;
    for (int state = 0; state <= RunSerial; ++state) {
      total += (double)dmp->timing[state].inround;
    }

    if (total == 0)
      continue;

    // Percentages.
    for (int state = 0; state <= RunSerial; ++state) {
      DmpThreadInfo::Timing* timing = &dmp->timing[state];

      const double t = (double)timing->inround / total;
      if (t > timing->max)
        timing->max = t;
      if (t < timing->min || timing->min == 0)
        timing->min = t;

      const double delta = t - timing->mean;
      timing->mean += delta / dmp->timing_total_quanta;
      // Use the new value of mean...
      timing->m2 += delta * (t - timing->mean);
    }
  }

  //
  // For each state, compute the spread of time each thread spent
  // in each that state.  Use this to update 'spread_mean' and
  // 'spread_m2' for each state.  The units are 'average time'.
  // For example, if three threads spend 1,2,3 ms in a given state
  // then the average time is '2 ms' and the spread is '1'.
  // A spread of 0 is ideal.
  //

  for (int state = 0; state <= RunSerial; ++state) {
    uint64_t t_max = 0;
    uint64_t t_min = 0;
    uint64_t t_total = 0;
    uint64_t threads = 0;

    for (int id = 0; id < DMPthreadInfosSize; ++id) {
      DmpThreadInfo* dmp = DMPthreadInfos[id];
      if (dmp->exited || dmp->state == Sleeping)
        continue;

      DmpThreadInfo::Timing* timing = &dmp->timing[state];
      threads++;

      const uint64_t t = timing->inround;
      if (t > t_max)
        t_max = t;
      if (t < t_min || t_min == 0)
        t_min = t;

      t_total += t;
    }

    if (t_max != 0 && threads > 1) {
      DmpThreadInfo::Timing* timing = &DMPthreadInfos[0]->timing[state];

      double total = (double)t_total / (double)threads;
      double temp_spread = (double)(t_max - t_min) / total;
      double delta_spread = temp_spread - timing->spread_mean;

      timing->spread_mean += delta_spread / DMPtimingRoundCount;
      // Use the new value of spread_mean...
      timing->spread_m2 += delta_spread * (temp_spread - timing->spread_mean);
    }
  }


  //
  // Reset!
  //

  for (int id = 0; id < DMPthreadInfosSize; ++id) {
    for (int state = 0; state <= RunSerial; ++state) {
      DMPthreadInfos[id]->timing[state].inround = 0;
    }
  }
}

#endif  // DMP_ENABLE_QUANTUM_TIMING

#ifdef DMP_ENABLE_ROUND_TIMING

struct DmpRoundTime {
  uint64_t total;
  uint64_t last_tsc;
};

DmpRoundTime DMPparallelModeTime;
DmpRoundTime DMPcommitModeTime;
DmpRoundTime DMPserialModeTime;

void DMProundTimeTransition(DmpRoundTime* from, DmpRoundTime* to) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const uint64_t now = timespec_to_nanoseconds(&ts);
  // Stop 'from'.
  if (from->last_tsc != 0) {
    if (DMPnumRunnableThreads >= DMP_NUM_PHYSICAL_PROCESSORS) {
      from->total += now - from->last_tsc;
    }
    from->last_tsc = 0;
  }
  // Start 'to'.
  to->last_tsc = now;
}

#endif  // DMP_ENABLE_ROUND_TIMING

//--------------------------------------------------------------
// Wrapper: DMP_resetRound
//--------------------------------------------------------------

#ifdef DMP_resetRound
#undef DMP_resetRound
extern void __DMP_resetRound__(const int oldRoundSync);

void DMP_resetRound(const int oldRoundSync) {
#ifdef DMP_ENABLE_QUANTUM_TIMING
  if (!DMPQuantumEndedWithAcquire && DMPnumRunnableThreads > 1) {
    update_quantum_timings();
  }
  DMPMAP->timing_last_tsc = 0;
  DMPQuantumEndedWithAcquire = false;
#endif
  __DMP_resetRound__(oldRoundSync);
}

#endif  // DMP_resetRound

//--------------------------------------------------------------
// Wrapper: DMP_setState
//--------------------------------------------------------------

#ifdef DMP_setState
#undef DMP_setState

void DMP_setState(DmpThreadInfo* dmp, const DmpThreadState s) {
#ifdef DMP_ENABLE_INSTRUMENT_WORK
  if (DMPnumRunnableThreads > 1) {
    uint64_t c = DMP_SCHEDULING_CHUNK_SIZE - DMPMAP->schedulingChunk;
    if (c > DMPMAP->work_this_quantum) {
      uint64_t w = c - DMPMAP->work_this_quantum;
      DMPMAP->work[dmp->state] += w;
      DMPMAP->work_this_quantum += w;
    }
    if (c == 0) {
      DMPMAP->work_this_quantum = 0;
    }
  }
#endif
#ifdef DMP_ENABLE_QUANTUM_TIMING
  {
    // TSC is too unstable, so we're using CLOCK_MONOTONIC (slower!).
#if 0
    const uint64_t now = rdtsc();
#else
    struct timespec ts;
    clock_gettime(DMP_QUANTUM_TIMING_CLOCK, &ts);
    const uint64_t now = timespec_to_nanoseconds(&ts);
#endif
    if (DMPMAP->timing_last_tsc != 0) {
      DMPMAP->timing[dmp->state].inround += now - DMPMAP->timing_last_tsc;
    }
    DMPMAP->timing_last_tsc = now;
  }
#endif
  __DMP_setState__(dmp, s);
}

#endif  // DMP_setState

//-----------------------------------------------------------------------
// Stats printing
//-----------------------------------------------------------------------

static const char* threadStateString(const DmpThreadState state);

#ifdef DMP_ENABLE_INSTRUMENT_ACQUIRES
static void printOwnershipRecord(FILE* f, OwnershipRecord* r) {
  if (r->nhistory == 0)
    return;
  fprintf(f, "  0x%x: [", r->id);
  const char* sep = "";
  int i;
  for (i = 0; i < r->nhistory; ++i) {
    fprintf(f, "%s0x%x", sep, r->history[i]);
    sep = ", ";
  }
  fprintf(f, "],\n");
}
#endif

static void printStat(DmpThreadInfo* dmp, const char* name, uint64_t stat) {
  printf("{'DMP':None,'ThreadID':%d,'%s':%llu}\n", dmp->threadID, name, stat);
}

static void printStatf(DmpThreadInfo* dmp, const char* name, double stat) {
  printf("{'DMP':None,'ThreadID':%d,'%s':%f}\n", dmp->threadID, name, stat);
}

void DMPinstrumentation_print_thread_statistics(struct DmpThreadInfo* dmp) {
  if (dmp->threadID == 0) {
    INFO_MSG("MOT Granularity: %d", DMP_MOT_GRANULARITY);
  }

#ifdef DMP_ENABLE_ROUND_TIMING
  if (dmp->threadID == 0) {
    uint64_t total = DMPparallelModeTime.total + DMPcommitModeTime.total +
                     DMPserialModeTime.total;
    printStatf(dmp, "ParallelModePercent",
        100.0 * ((double)DMPparallelModeTime.total / (double)total));
    printStatf(dmp, "CommitModePercent",
        100.0 * ((double)DMPcommitModeTime.total / (double)total));
    printStatf(dmp, "SerialModePercent",
        100.0 * ((double)DMPserialModeTime.total / (double)total));
  }
#endif

#ifdef DMP_ENABLE_QUANTUM_TIMING
  if (dmp->threadID == 0) {
    for (int s = 0; s < RunSerial+1; ++s) {
      DmpThreadInfo::Timing* t = &dmp->timing[s];
      const char* name = threadStateString((DmpThreadState)s);
      double spread_variance = t->spread_m2 / (double)DMPtimingRoundCount;
      printf("%s Normalized-Spread Mean: %f\n", name, t->spread_mean);
      printf("%s Normalized-Spread StdDev: %f\n", name, sqrt(spread_variance));
    }
    struct timespec ts;
    clock_getres(DMP_QUANTUM_TIMING_CLOCK, &ts);
    printf("Timer resolution: %llu ns\n", timespec_to_nanoseconds(&ts));
  }

  printf("Thread %d NQuanta: %d\n", dmp->threadID, dmp->timing_total_quanta);
  for (int s = 0; s < RunSerial+1; ++s) {
    DmpThreadInfo::Timing* t = &dmp->timing[s];
    const char* name = threadStateString((DmpThreadState)s);
    double variance = t->m2 / dmp->timing_total_quanta;
    printf("Thread %d %s %%-time Max: %f\n", dmp->threadID, name, t->max);
    printf("Thread %d %s %%-time Min: %f\n", dmp->threadID, name, t->min);
    printf("Thread %d %s %%-time Mean: %f\n", dmp->threadID, name, t->mean);
    printf("Thread %d %s %%-time StdDev: %f\n", dmp->threadID, name, sqrt(variance));
  }
#endif

#ifdef DMP_ENABLE_INSTRUMENT_ACQUIRES
  if (dmp->threadID == 0) {
    INFO_MSG("Rounds: %d", DMProundNumber);
    INFO_MSG("RoundsWithSharedMutexes: %d", DMProundsWithSharedMutexes);

    // Dump the ownership histories to a file.
    FILE* f;
    const char* dir = getenv("DMP_HOME_DIR");
    if (dir) {
      char filename[1024];
      snprintf(filename, sizeof filename, "%s/owner-history.txt", dir);
      f = fopen(filename, "w");
    } else {
      f = fopen("owner-history.txt", "w");
    }
    if (f == NULL) {
      printf("failed to open dmhistories.py!\n");
    } else {
      int i;
      fprintf(f, "dataHistories = {\n");
      for (i = 0; i < HistorySize; ++i) {
        printOwnershipRecord(f, dataHistories + i);
      }
      fprintf(f, "}\n");
      fprintf(f, "mutexHistories = {\n");
      for (i = 0; i < HistorySize; ++i) {
        printOwnershipRecord(f, resourceHistories + i);
      }
      fprintf(f, "}\n");
      fclose(f);
    }
  }
#endif

#ifdef DMP_ENABLE_INSTRUMENT_WORK
  if (dmp->threadID == 0) {
    uint64_t work[RunSerial+1];
    uint64_t t_work = 0;
    uint64_t t_threads = 0;
    uint64_t t_toserial_total = 0;
    uint64_t t_toserial_excall = 0;
    uint64_t t_toserial_mb = 0;
    uint64_t t_toserial_mbresource = 0;
    memset(work, 0, sizeof work);
#ifdef DMP_ENABLE_BUFFERED_MODE
    uint64_t t_wb_quanta = 0;
    uint64_t t_wb_size = 0;
    uint64_t t_wb_maxsize = 0;
    uint64_t t_wb_used = 0;
    uint64_t t_wb_maxused = 0;
    uint64_t t_wb_maxhashchain = 0;
    uint64_t t_wb_totalhashbuckets = 0;
    uint64_t t_wb_totalcommitslocked = 0;
    uint64_t t_wb_synctotal = 0;         // HB_SYNC only
    uint64_t t_wb_syncwithoutwait = 0;   // HB_SYNC only
#endif
    // Summarize.
    for (int i = 0; i < DMPthreadInfosSize; ++i) {
      DmpThreadInfo* dmp = DMPthreadInfos[i];
      for (int s = 0; s < RunSerial+1; ++s) {
        work[s] += dmp->work[s];
        t_work += dmp->work[s];
      }
      t_threads++;
      t_toserial_total  += dmp->toserial_total;
      t_toserial_excall += dmp->toserial_excall;
      t_toserial_mb     += dmp->toserial_mb;
      t_toserial_mbresource += dmp->toserial_mbresource;
#ifdef DMP_ENABLE_BUFFERED_MODE
      t_wb_quanta += dmp->wb_totalquanta;
      t_wb_size += dmp->wb_totalsize;
      t_wb_used += dmp->wb_totalused;
      t_wb_totalhashbuckets += dmp->wb_totalhashbuckets;
      t_wb_totalcommitslocked += dmp->wb_totalcommitslocked;
      t_wb_synctotal       += dmp->wb_synctotal;
      t_wb_syncwithoutwait += dmp->wb_syncwithoutwait;
      if (dmp->wb_maxsize > t_wb_maxsize)
        t_wb_maxsize = dmp->wb_maxsize;
      if (dmp->wb_maxused > t_wb_maxused)
        t_wb_maxused = dmp->wb_maxused;
      if (dmp->wb_maxhashchain > t_wb_maxhashchain)
        t_wb_maxhashchain = dmp->wb_maxhashchain;
#endif
    }
    // Output.
    for (int s = 0; s < RunSerial+1; ++s) {
      char stat[512];
      snprintf(stat, sizeof stat, "TotalWork%s", threadStateString((DmpThreadState)s));
      if (t_work == 0) {
        printStat(dmp, stat, 0);
      } else {
        double pct = ((double)work[s] / (double)(t_work)) * 100.0;
        printStatf(dmp, stat, pct);
      }
    }
    printStatf(dmp, "TotalToSerialForExternalCall",
        (t_toserial_total == 0)
          ? 0 : ((double)t_toserial_excall/(double)t_toserial_total) * 100.0);
    printStatf(dmp, "TotalToSerialForMemBarrier",
        (t_toserial_total == 0)
          ? 0 : ((double)t_toserial_mb/(double)t_toserial_total) * 100.0);
    printStatf(dmp, "TotalToSerialForResource",
        (t_toserial_total == 0)
          ? 0 : ((double)t_toserial_mbresource/(double)t_toserial_total) * 100.0);
#ifdef DMP_ENABLE_BUFFERED_MODE
    printStatf(dmp, "AvgWriteBufferCommitsLocked",
        (t_wb_size == 0) ? 0 : ((double)t_wb_totalcommitslocked/(double)t_wb_size));
    printStatf(dmp, "AvgWriteBufferEntries",
        (t_wb_quanta == 0) ? 0 : ((double)t_wb_size/(double)t_wb_quanta));
    printStatf(dmp, "AvgWriteBufferBytesUsed",
        (t_wb_quanta == 0) ? 0 : ((double)t_wb_used/(double)t_wb_quanta));
    printStat(dmp, "MaxWriteBufferEntries", t_wb_maxsize);
    printStat(dmp, "MaxWriteBufferBytesUsed", t_wb_maxused);
    printStat(dmp, "OverallWriteBufferMaxChainLength", t_wb_maxhashchain);
    printStatf(dmp, "OverallWriteBufferAvgBucketsUsed",
        (t_wb_quanta == 0) ?
          0 : (double)t_wb_totalhashbuckets / (double)(t_threads * t_wb_quanta));
    printStat(dmp, "TotalSyncs", t_wb_synctotal);
    printStat(dmp, "TotalSyncsWithoutWait", t_wb_syncwithoutwait);
#endif
  }
  {
    uint64_t t_work = 0;
    for (int s = 0; s < RunSerial+1; ++s)
      t_work += dmp->work[s];
    printStat(dmp, "SerialWork",
      (t_work == 0) ? 0 : ((double)dmp->work[RunSerial] / (double)t_work) * 100.0);
  }
  printStatf(dmp, "ToSerialForExternalCall",
      (dmp->toserial_total == 0)
        ? 0 : ((double)dmp->toserial_excall/(double)dmp->toserial_total) * 100.0);
  printStatf(dmp, "ToSerialForMemBarrier",
      (dmp->toserial_total == 0)
        ? 0 : ((double)dmp->toserial_mb/(double)dmp->toserial_total) * 100.0);
  printStatf(dmp, "ToSerialForResource",
      (dmp->toserial_total == 0)
        ? 0 : ((double)dmp->toserial_mbresource/(double)dmp->toserial_total) * 100.0);
#ifdef DMP_ENABLE_BUFFERED_MODE
  printStat(dmp, "WriteBufferHashMaxChainLength", dmp->wb_maxhashchain);
  printStatf(dmp, "WriteBufferHashAvgChainLength",
      (dmp->wb_totalhashbuckets == 0) ?
        0 : (double)dmp->wb_totalhashchains / (double)dmp->wb_totalhashbuckets);
  printStatf(dmp, "WriteBufferHashAvgBucketsUsed",
      (dmp->wb_totalquanta == 0) ?
        0 : (double)dmp->wb_totalhashbuckets / (double)dmp->wb_totalquanta);
  printStat(dmp, "Syncs", dmp->wb_synctotal);
  printStat(dmp, "SyncsWithoutWait", dmp->wb_syncwithoutwait);
#endif
#endif

#ifdef DMP_ENABLE_MODEL_STM
  if (dmp->threadID == 0) {
    extern uint64_t DMPstmParallelQuanta;
    extern uint64_t DMPstmSerialQuanta;
    uint64_t total = DMPstmParallelQuanta + DMPstmSerialQuanta;
    printStatf(dmp, "ParallelModePercent",
        100.0 * ((double)DMPstmParallelQuanta / (double)total));
    printStatf(dmp, "CommitModePercent",
        100.0 * 0);
    printStatf(dmp, "SerialModePercent",
        100.0 * ((double)DMPstmSerialQuanta / (double)total));
  }
#endif

  fflush(stdout);
}

//-----------------------------------------------------------------------
// Debugging
//-----------------------------------------------------------------------

#ifdef DMP_ENABLE_MODEL_O_B_S
extern atomic_uint_t DMPbufferingBarrier;
#endif
#ifdef DMP_ENABLE_BUFFERED_MODE
extern atomic_uint_t DMPcommitBarrier;
#endif
extern atomic_uint_t DMPserialBarrier;
extern atomic_uint_t DMProundBarrier;
extern atomic_uint_t DMPscheduledThread;

#define DEBUG_THREAD_PTR(t) ((t) ? (t)->threadID : -1), (t)
#define DEBUG_RESOURCE_PTR(r) resourceTypeString(r), (r)

static const char* threadStateString(const DmpThreadState state) {
  switch (state) {
    case Sleeping: return "Sleeping";
    case JustWokeUp: return "JustWokeUp";
#ifdef DMP_ENABLE_OWNERSHIP_MODE
    case WaitForOwnership: return "WaitForOwnership";
    case RunOwnership: return "RunOwnership";
#endif
#ifdef DMP_ENABLE_BUFFERED_MODE
    case WaitForBuffered: return "WaitForBuffered";
    case RunBuffered: return "RunBuffered";
    case WaitForCommit: return "WaitForCommit";
    case RunCommit: return "RunCommit";
#endif
    case WaitForSerial: return "WaitForSerial";
    case WaitForSerialToken: return "WaitForSerialToken";
    case RunSerial: return "RunSerial";
    default: return "???";
  }
}

static const char* resourceTypeString(DMPresource* r) {
  if (!r) return "nil";
  switch (DMPresource_type(r)) {
    case DMP_RESOURCE_TYPE_MUTEX: return "mutex";
    case DMP_RESOURCE_TYPE_RWLOCK: return "rwlock";
    case DMP_RESOURCE_TYPE_SEM: return "sem";
    case DMP_RESOURCE_TYPE_CONDVAR: return "condvar";
    case DMP_RESOURCE_TYPE_BARRIER: return "barrier";
    case DMP_RESOURCE_TYPE_ONCE: return "once";
    default: return "???";
  }
}

__attribute__((used))
void DMPprintThreadInfo(DmpThreadInfo* dmp) {
  printf("THREAD%2d: %p\n"
         "          (state %s) (blockedOn %s:%p) (resourceNesting %d) (chunk %d of %d)\n"
         "          (canceled %d) (exited %d) (joiner %d/%p) (holds %s:%p/%s:%p)\n"
         "          (prev %d/%p) (next %d/%p)\n",
         dmp->threadID, dmp,
         threadStateString(dmp->state),
#ifdef DMP_ENABLE_FAST_HANDOFF
         DEBUG_RESOURCE_PTR(dmp->blockedOn),
#else
         DEBUG_RESOURCE_PTR(NULL),
#endif
#ifdef DMP_ENABLE_TINY_SERIAL_MODE
         dmp->resourceNesting,
#else
         0,
#endif
         dmp->schedulingChunk, DMP_SCHEDULING_CHUNK_SIZE,
         dmp->canceled, dmp->exited,
         DEBUG_THREAD_PTR(dmp->joiner),
#ifdef DMP_ENABLE_DATA_GROUPING
         DEBUG_RESOURCE_PTR(dmp->innerResource),
         DEBUG_RESOURCE_PTR(dmp->nextResource),
#else
         DEBUG_RESOURCE_PTR(NULL),
         DEBUG_RESOURCE_PTR(NULL),
#endif
         DEBUG_THREAD_PTR(dmp->prevRunnable),
         DEBUG_THREAD_PTR(dmp->nextRunnable));
}

__attribute__((used))
void DMPprintScheduler() {
  printf("SCHED: (round %llu) (runnable %d) (live %d)",
         DMProundNumber, DMPnumRunnableThreads, DMPnumLiveThreads);
#if defined(DMP_ENABLE_MODEL_O_B_S)
  printf(" (bufferB %d) (commitB %d) (serialB %d) (roundB %d) (serialsched %d)\n",
         DMPbufferingBarrier, DMPcommitBarrier, DMPserialBarrier, DMProundBarrier,
         DMPscheduledThread);
#elif defined(DMP_ENABLE_BUFFERED_MODE)
  printf(" (commitB %d) (serialB %d) (roundB %d) (serialsched %d)\n",
         DMPcommitBarrier, DMPserialBarrier, DMProundBarrier, DMPscheduledThread);
#else
  printf(" (serialB %d) (roundB %d) (serialsched %d)\n",
         DMPserialBarrier, DMProundBarrier, DMPscheduledThread);
#endif
  printf("==RUNNABLE==\n");
  int printed[MaxThreads];
  memset(printed, 0, sizeof printed);
  for (DmpThreadInfo* dmp = DMPfirstRunnable; dmp; dmp = dmp->nextRunnable) {
    if (printed[dmp->threadID])
      break;
    DMPprintThreadInfo(dmp);
    printed[dmp->threadID] = 1;
  }
  int done = 1;
  for (int i = 0; i < DMPthreadInfosSize; ++i)
    if (!printed[i]) {
      done = 0;
      break;
    }
  if (done)
    return;
  printf("==SLEEPING==\n");
  for (int i = 0; i < DMPthreadInfosSize; ++i)
    if (!printed[i])
      DMPprintThreadInfo(DMPthreadInfos[i]);
}

//
// Code Location Debugging
// When built with -dmp-memtrack-dbg-code-loc, LLVM inserts
// calls to DMPsetCodeLocation() just before every memory access.
//

void DMPsetCodeLocation(int loc) {
  __sync_synchronize();
  DMPMAP->codeLocation = loc;
  __sync_synchronize();
}

__attribute__((used))
void DMPprintCodeLocationForThread(int id) {
  printf("Thread %d code location: %d\n", id, DMPthreadInfos[id]->codeLocation);
}

__attribute__((used))
void DMPprintCodeLocation() {
  printf("Thread %d code location: %d\n", DMPMAP->threadID, DMPMAP->codeLocation);
}
