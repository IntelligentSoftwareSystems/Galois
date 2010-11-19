/**
 * @file hoarddetours.cpp
 * @brief Hooks to Hoard for Detours.
 * @author Emery Berger <http://www.cs.umass.edu/~emery>
 */

/*
   To compile, run compile-detours.cmd, or do the following:

   cl /I.. /LD /MD /DNDEBUG /Ox /Zp8 /Oa /G6 /Oy /I/detours/include hoarddetours.cpp  \detours\lib\detours.lib /link /subsystem:console /entry:_DllMainCRTStartup@12 /force:multiple

   To use (via withdll.exe, from Detours):

   detours\withdll -d:hoarddetours.dll myexecutable.exe

  */

#include "VERSION.h"

#define CUSTOM_PREFIX(n) hoard##n

#if !defined(_WIN32)
#error "This file is for Windows only."
#endif

#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0500
#define NT

#include <windows.h>
#include <stdio.h>
#include <malloc.h>
#include "detours.h"

#if defined(_WIN32)
#pragma inline_depth(255)
#endif

#include "cpuinfo.h"
#include "hoard.h"

volatile int anyThreadCreated = 0;

const int MaxThreads = 512;
const int NumHeaps = 64;

using namespace Hoard;

class TheCustomHeapType : public HoardHeap<MaxThreads, NumHeaps> {};

inline static TheCustomHeapType * getCustomHeap (void) {
  static char thBuf[sizeof(TheCustomHeapType)];
  static TheCustomHeapType * th = new (thBuf) TheCustomHeapType;
  return th;
}

// Allocate exactly one object.
static void * objectAllocated = (void *) 0x1;

extern "C" {
  DETOUR_TRAMPOLINE(void __cdecl real_onexit (void (*f)(void)),
		    onexit);

  DETOUR_TRAMPOLINE(void __cdecl real_exit (int code),
		    exit);

  DETOUR_TRAMPOLINE(void *  __cdecl real_malloc (size_t sz),
		    malloc);
  
  DETOUR_TRAMPOLINE(void *  __cdecl real_realloc (void * ptr, size_t sz),
		    realloc);
  
  DETOUR_TRAMPOLINE(void *  __cdecl real_calloc (size_t sz, size_t n),
		    calloc);
  
  DETOUR_TRAMPOLINE(void *  __cdecl real_free (void * ptr),
		    free);
  
  DETOUR_TRAMPOLINE(size_t  __cdecl real_msize (void * ptr),
		    _msize);

  DETOUR_TRAMPOLINE(void * __cdecl real_expand(void * memblock, size_t size),
		    _expand);

  DETOUR_TRAMPOLINE(char *  __cdecl real_strdup (const char * s),
		    strdup);

}

/* Detours */

#include "wrapper.cpp"

// We use Hoard to manage small objects and the original allocator
// to manage large objects.

static const int HOARD_MAX_EXIT_FUNCTIONS = 512;
static int exitCount = 0;

typedef void (*exitFunctionType) (void);
exitFunctionType exitFunctionBuffer[HOARD_MAX_EXIT_FUNCTIONS];

extern "C" void my_onexit (void (*function)(void)) {
  if (exitCount < HOARD_MAX_EXIT_FUNCTIONS) {
    exitFunctionBuffer[exitCount] = function;
    exitCount++;
  }
}

extern "C" void my_exit (int code) {
  while (exitCount > 0) {
    exitCount--;
    (exitFunctionBuffer[exitCount])();
  }
  real_exit (code);
}

extern "C" void my_cexit (void) {
  my_exit (0);
}

extern "C" void * my_expand (void * memblock, size_t size)
{
  // Disable by always returning NULL (the error condition for _expand).
  return NULL;
}


/* Install Detours trampolines. */

#define REPLACE(module,x,fn) \
{ \
  PBYTE p; \
  p = DetourFindFunction ((module), (x)); \
  if (p) { \
    DetourFunction (p, (PBYTE) (fn));  \
  } \
}

#define REMOVE(module,x,fn) \
{ \
  PBYTE p; \
  p = DetourFindFunction ((module), (x)); \
  if (p) { \
    DetourRemove (p, (PBYTE) (fn));  \
  } \
}


void trampolineInstall (HMODULE hModule,
			bool insert)
{
  typedef PBYTE (__stdcall *drtype)(PBYTE, PBYTE);
  typedef BOOL (__stdcall *dfwtype)(PBYTE, PBYTE);
  PBYTE p;
  drtype dr;
  dfwtype dfw;

  if (insert) {
    dfw = &DetourFunctionWithTrampoline;
    dr = &DetourFunction;
  } else {
    dfw = &DetourRemove;
    dr = (drtype) &DetourRemove;
  }

  (*dfw) ((PBYTE) real_realloc, (PBYTE) CUSTOM_PREFIX(realloc));
  (*dfw) ((PBYTE) real_calloc,  (PBYTE) CUSTOM_PREFIX(calloc));
  (*dfw) ((PBYTE) real_msize,   (PBYTE) CUSTOM_PREFIX(malloc_usable_size));
  (*dfw) ((PBYTE) real_strdup,  (PBYTE) CUSTOM_PREFIX(strdup));
  (*dfw) ((PBYTE) real_malloc,  (PBYTE) CUSTOM_PREFIX(malloc));
  (*dfw) ((PBYTE) real_free,    (PBYTE) CUSTOM_PREFIX(free));
  (*dfw) ((PBYTE) real_expand,  (PBYTE) my_expand);
  (*dfw) ((PBYTE) real_onexit,  (PBYTE) my_onexit);
  (*dfw) ((PBYTE) real_exit,  (PBYTE) my_exit);

  char dllName[MAX_PATH];

  for (HINSTANCE hInst = NULL; (hInst = DetourEnumerateModules(hInst)) != NULL; ) {
    GetModuleFileName(hInst, dllName, MAX_PATH);

    // Replace the memory allocation calls in the MSVCR library.
   
    if ((strstr(dllName, "MSVCR") != 0) || (strstr(dllName, "msvcr") != 0)) {

      // operator new, new[], delete, delete[]
      // see undname

      // operator new
      p = DetourFindFunction (dllName, "??2@YAPAXI@Z");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(malloc));

      // operator new[]
      p = DetourFindFunction (dllName, "??_U@YAPAXI@Z");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(malloc));
      
      // operator delete
      p = DetourFindFunction (dllName, "??3@YAXPAX@Z");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(free));

      // operator delete[]
      p = DetourFindFunction (dllName, "??_V@YAXPAX@Z");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(free));

      p = DetourFindFunction (dllName, "realloc");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(realloc));
      
      p = DetourFindFunction (dllName, "calloc");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(calloc));
      
      p = DetourFindFunction (dllName, "_msize");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(malloc_usable_size));
      
      p = DetourFindFunction (dllName, "strdup");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(strdup));
      
      p = DetourFindFunction (dllName, "malloc");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(malloc));
      
      p = DetourFindFunction (dllName, "free");
      if (p) (*dr)(p, (PBYTE) CUSTOM_PREFIX(free));
      
      p = DetourFindFunction (dllName, "_expand");
      if (p) (*dr)(p, (PBYTE) my_expand);

      p = DetourFindFunction (dllName, "_onexit");
      if (p) (*dr)(p, (PBYTE) my_onexit);

      p = DetourFindFunction (dllName, "exit");
      if (p) (*dr)(p, (PBYTE) my_exit);

      p = DetourFindFunction (dllName, "_cexit");
      if (p) (*dr)(p, (PBYTE) my_cexit);
    }
  }
}

void trampolineWith (HMODULE hModule) {
  trampolineInstall (hModule, true);
}


#if defined(_WIN32)
#pragma warning(disable:4273)
#endif

static void initializeMaps (void) {
  int i;
  for (i = 0; i < TheCustomHeapType::MaxThreads; i++) {
    getCustomHeap()->setTidMap(i, 0);
  }
  for (i = 0; i < TheCustomHeapType::MaxHeaps; i++) {
    getCustomHeap()->setInusemap (i, 0);
  }
}

static void findUnusedHeap (void) {
  // Find an unused heap.
  
  int tid = HL::CPUInfo::getThreadId() % TheCustomHeapType::MaxThreads;

  int i = 0;
  while ((i < TheCustomHeapType::MaxHeaps) && (getCustomHeap()->getInusemap(i)))
    i++;
  if (i >= TheCustomHeapType::MaxHeaps) {
    // Every heap is in use: pick a random victim.
    i = (int) ( TheCustomHeapType::MaxHeaps * ((double) rand() / (double) RAND_MAX));
  }
  int v = getCustomHeap()->getInusemap(i);

  getCustomHeap()->setInusemap (i, 1);
  getCustomHeap()->setTidMap (tid, i);
}

static void releaseHeap (void) {
  // Decrement the ref-count on the current heap.

  enum { VerifyPowerOfTwo = 1 / ((TheCustomHeapType::MaxThreads & ~(TheCustomHeapType::MaxThreads-1))) };

  int tid = HL::CPUInfo::getThreadId() & (TheCustomHeapType::MaxThreads - 1);
  int heapIndex = getCustomHeap()->getTidMap (tid);

  //  printf ("thread %d releasing heap %d\n", tid, heapIndex);
 
  int v = getCustomHeap()->getInusemap (heapIndex);
  getCustomHeap()->setInusemap (heapIndex, 0);

  // Prevent underruns (defensive programming).

  if (getCustomHeap()->getInusemap (heapIndex) < 0) {
    getCustomHeap()->setInusemap (heapIndex, 0);
  }
}


BOOL APIENTRY DllMain (HINSTANCE hModule,
		       DWORD dwReason,
		       PVOID lpReserved)
{
  int i;
  int tid;
  static int np = HL::CPUInfo::computeNumProcessors();
  char dllName[MAX_PATH];
  switch (dwReason) {
  case DLL_PROCESS_ATTACH:
    fprintf (stderr, "This software uses the Hoard scalable memory allocator (version " HOARD_VERSION_STRING ", Detours).\nCopyright (C) 2005 Emery Berger, The University of Texas at Austin,\nand the University of Massachusetts Amherst.\nFor more information, see http://www.hoard.org\n");
    initializeMaps();
    trampolineWith (hModule);
    break;
  case DLL_PROCESS_DETACH:
    // Notice that we haven't replaced all heap calls. Here's one.
    objectAllocated = (void *) real_malloc(1);
    break;
  case DLL_THREAD_ATTACH:
    anyThreadCreated = 1;
    if (np == 1) {
      getCustomHeap()->setTidMap ((int) HL::CPUInfo::getThreadId() % TheCustomHeapType::MaxThreads, 0);
    } else {
      findUnusedHeap();
    }
    break;
  case DLL_THREAD_DETACH:
    if (np != 1) {
      releaseHeap();
    }
    break;
  }
  return TRUE;
}


