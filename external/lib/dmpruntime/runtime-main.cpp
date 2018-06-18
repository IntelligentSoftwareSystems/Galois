// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Runtime - main/exit
//

#include "dmp-internal.h"
#include "dmp-internal-mot.h"
#include "dmp-internal-wb.h"
#include <dlfcn.h>

// Constants.
#define DefaultSchedulingChunkSize 1000
#define DefaultNumPhysicalProcessors 8

int DMP_SCHEDULING_CHUNK_SIZE   = DefaultSchedulingChunkSize;
int DMP_NUM_PHYSICAL_PROCESSORS = DefaultNumPhysicalProcessors;

// Global data.
DmpThreadInfo* DMPfirstRunnable    = NULL;
atomic_int_t DMPfirstRunnableID    = 0;
atomic_int_t DMPnumRunnableThreads = 0;
atomic_int_t DMPnumLiveThreads     = 0;
atomic_int_t DMPthreadInfosSize    = 0;
DmpThreadInfo* DMPthreadInfos[MaxThreads];

thread_local DmpThreadInfo* DMPMAP;

// The MOT.
uint32_t DMPmot[DMP_MOT_ENTRIES];

#ifdef DMP_ENABLE_WB_HBSYNC
// Global work counters.
cacheline_padded_int32_t DMPglobalSchedulingChunks[MaxThreads];
#endif

//-----------------------------------------------------------------------
// Allocators
//-----------------------------------------------------------------------

int (*real_posix_memalign)(void**, size_t, size_t);
void* (*real_malloc)(size_t);
void* (*real_realloc)(void*, size_t);
void (*real_free)(void*);

static void init_allocators() {
  // When linking with libhoard, DMP code must always call the underlying
  // allocator to avoid circular dependencies between libhoard and DMP.
  real_posix_memalign =
      (__typeof__(real_posix_memalign))dlsym(RTLD_NEXT, "posix_memalign");
  real_malloc  = (__typeof__(real_malloc))dlsym(RTLD_NEXT, "malloc");
  real_realloc = (__typeof__(real_realloc))dlsym(RTLD_NEXT, "realloc");
  real_free    = (__typeof__(real_free))dlsym(RTLD_NEXT, "free");
}

void* alloc_cache_aligned(size_t size) {
  size = ROUNDUP_MOD_N(size, CACHELINE_SIZE);
  void* addr;
  if (real_posix_memalign(&addr, CACHELINE_SIZE, size) != 0)
    ERROR_MSG("alloc_cache_aligned failed\n");
  memset(addr, 0, size);
  return addr;
}

//-----------------------------------------------------------------------
// Quantum Boundaries
//-----------------------------------------------------------------------

void DMPcommit(int chunkSize) {
  DMPMAP->schedulingChunk -= chunkSize;
  if (unlikely(DMPMAP->schedulingChunk <= 0))
    DMP_waitForNextQuantum();
#ifdef DMP_ENABLE_WB_HBSYNC
  else if (unlikely(DMPMAP->schedulingChunk <=
                    DMPMAP->triggerGlobalSchedulingChunkUpdate)) {
    DMPglobalSchedulingChunks[DMPMAP->threadID].val = DMPMAP->schedulingChunk;
    DMPMAP->triggerGlobalSchedulingChunkUpdate      = 0;
  }
#endif
}

void DMPprepareForExternalCall(int chunkSize) {
  // Force into serial mode to execute the external call.
  // Note that we won't "commit" until the next DMPcommit.
#ifdef DMP_ENABLE_TINY_SERIAL_MODE
  DMPMAP->schedulingChunk = 100;
#else
  DMPMAP->schedulingChunk -= chunkSize;
#endif
#ifdef DMP_ENABLE_INSTRUMENT_WORK
  if (DMPMAP->state != RunSerial)
    DMPMAP->toserial_excall++;
#endif
  DMP_waitForSerialMode();
}

void DMPmembarrier(void) {
  // Called before an explicit memory barrier.
  // We cannot buffer across this call.

#if defined(DMP_ENABLE_MODEL_O_S) || defined(DMP_ENABLE_MODEL_STM)
  // Nop

#elif defined(DMP_ENABLE_MODEL_B_S) || defined(DMP_ENABLE_MODEL_OB_S)
#ifdef DMP_ENABLE_INSTRUMENT_WORK
  if (DMPMAP->state != RunSerial)
    DMPMAP->toserial_mb++;
#endif
#ifdef DMP_ENABLE_TINY_SERIAL_MODE
  DMP_waitForNextQuantum();
#else
  DMP_waitForSerialMode();
#endif

#elif defined(DMP_ENABLE_MODEL_O_B_S)
  if (DMPMAP->state == RunBuffered) {
#ifdef DMP_ENABLE_INSTRUMENT_WORK
    DMPMAP->toserial_mb++;
#endif
#ifdef DMP_ENABLE_TINY_SERIAL_MODE
    DMP_waitForNextQuantum();
#else
    DMP_waitForSerialMode();
#endif
  }
#endif
}

void DMPmembarrierResource(void) {
  // Called before using a shared resource.
  // If buffering, we must enter serial mode.

#if defined(DMP_ENABLE_MODEL_O_S) || defined(DMP_ENABLE_MODEL_STM)
  // Nop

#elif defined(DMP_ENABLE_MODEL_B_S) || defined(DMP_ENABLE_MODEL_OB_S)
#ifdef DMP_ENABLE_INSTRUMENT_WORK
  if (DMPMAP->state != RunSerial)
    DMPMAP->toserial_mbresource++;
#endif
  DMP_waitForSerialMode();

#elif defined(DMP_ENABLE_MODEL_O_B_S)
  if (DMPMAP->state == RunBuffered) {
#ifdef DMP_ENABLE_INSTRUMENT_WORK
    DMPMAP->toserial_mbresource++;
#endif
    DMP_waitForSerialMode();
  }
#endif
}

//-----------------------------------------------------------------------
// Indirect calls
//-----------------------------------------------------------------------

// NULL-terminated list
extern char* DMPinstrumentedFunctionsList[];

// Hash table
struct TableNode {
  char* functions[4];
  TableNode* next;
};
static TableNode* DMPinstrumentedFunctionsTable[256];

static inline int functionPtrHash(const char* p) {
  return (int)(((uintptr_t)p >> 6) & 255);
}

static void DMPinitInstrumentedFunctionsTable() {
  // Populate the hash table from the list.
  for (char** f = DMPinstrumentedFunctionsList; *f != NULL; ++f) {
    char* p        = *f;
    const int hash = functionPtrHash(p);
    TableNode** n  = &DMPinstrumentedFunctionsTable[hash];
    // Look for an open slot.
    while (*n != NULL) {
      for (int i = 1; i < 4; ++i) {
        if ((*n)->functions[i] == NULL) {
          (*n)->functions[i] = p;
          goto next_function;
        }
      }
      n = &(*n)->next;
    }
    // Need to allocate a new node.
    *n = (TableNode*)real_malloc(sizeof **n);
    memset(*n, 0, sizeof **n);
    (*n)->functions[0] = p;
    // Next.
  next_function: /**/;
  }
}

void DMPprepareForIndirectCall(int chunkSize, void* fnaddr) {
  const int hash = functionPtrHash((char*)fnaddr);
  // Internal?
  for (TableNode* n = DMPinstrumentedFunctionsTable[hash]; n != NULL;
       n            = n->next) {
    for (int i = 0; i < 4; ++i) {
      if (n->functions[i] == fnaddr) {
        DMPcommit(chunkSize);
        return;
      }
    }
  }
  // External.
  DMPprepareForExternalCall(chunkSize);
}

//-----------------------------------------------------------------------
// Main/Exit
//-----------------------------------------------------------------------

void DMP_printf(const char* msg, ...) {
  printf("[DMP] ");
  va_list arg;
  va_start(arg, msg);
  vprintf(msg, arg);
  va_end(arg);
  fflush(stdout);
}

static void DMP_terminate() {
  // This is called right before exit().
  printf("{'DMP':None,'CalledPthreadCreate':%d}\n", DMPthreadInfosSize);

  struct timeval tv;
  gettimeofday(&tv, NULL);
  DMP_printf("EXIT DMP Thread: %d: End: %ld s + %ld ms\n", DMPMAP->threadID,
             tv.tv_sec, tv.tv_usec);
  DMP_printf("Exit called...\n");

  // Finalize statistics.
  for (int i = 0; i < DMPthreadInfosSize; ++i) {
    DMPinstrumentation_print_thread_statistics(DMPthreadInfos[i]);
  }
  fflush(stdout);
}

static int DMP_init_called;
extern "C" void DMP_init(void) {
  dmp_static_assert(MaxThreads < SHRT_MAX);

  INFO_MSG("[DMP] Invoking DMP_init()");
  init_allocators();

  // Read the environment.
  char* s;
  s                         = getenv("DMP_SCHEDULING_CHUNK_SIZE");
  DMP_SCHEDULING_CHUNK_SIZE = s ? atoi(s) : DefaultSchedulingChunkSize;

  s                           = getenv("DMP_NUM_PHYSICAL_PROCESSORS");
  DMP_NUM_PHYSICAL_PROCESSORS = s ? atoi(s) : DefaultNumPhysicalProcessors;

  // Initialize the runtime.
  DMP_initRuntime();
  DMPthread_initMainThread();
  DMPinitInstrumentedFunctionsTable();

  INFO_MSG("[DMP] Completed DMP_init()");
  DMP_init_called = 1;
}

int main(int argc, char* argv[]) {
  ASSERT(*((volatile int*)&DMP_init_called));

  // Run this at program termination.
  atexit(DMP_terminate);

  // Run main.
  INFO_MSG("[DMP] Scheduling Chunk: %d", DMP_SCHEDULING_CHUNK_SIZE);
  INFO_MSG("[DMP] Num Physical CPUs: %d", DMP_NUM_PHYSICAL_PROCESSORS);
#ifdef DMP_ENABLE_OWNERSHIP_MODE
  INFO_MSG("[DMP] MOT granularity: %d", DMP_MOT_GRANULARITY);
  INFO_MSG("[DMP] MOT hashsize: %d", DMP_MOT_ENTRIES);
#endif
#ifdef DMP_ENABLE_BUFFERED_MODE
  INFO_MSG("[DMP] WB granularity: %d", DMP_WB_GRANULARITY);
  INFO_MSG("[DMP] WB hashsize: %d", DMP_WB_HASHSIZE);
#endif
  INFO_MSG("[DMP] Calling main()...");

  const int main_result = DMPmain(argc, argv);
  INFO_MSG("[DMP] main() completed");
  return main_result;
}

#if 0
#ifdef GALOIS_CHANGE
namespace {
// Do initialization in code rather than depend on llvm pass
struct Init {
  Init() { DMP_init();  DMPmain(0, NULL); }
};
Init iii;
}
char* DMPinstrumentedFunctionsList[1] = {NULL};
void DMP_Galois_init() {

}
#endif
#endif
