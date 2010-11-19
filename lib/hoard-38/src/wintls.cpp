// -*- C++ -*-
// Windows TLS functions.

DWORD LocalTLABIndex;

#include <new>

static TheCustomHeapType * initializeCustomHeap (void)
{
  // Allocate a per-thread heap.
  TheCustomHeapType * heap;
  size_t sz = sizeof(TheCustomHeapType) + sizeof(double);
  void * mh = getMainHoardHeap()->malloc(sz);
  heap = new ((char *) mh) TheCustomHeapType (getMainHoardHeap());

  // Store it in the appropriate thread-local area.
  TlsSetValue (LocalTLABIndex, heap);

  return heap;
}

inline TheCustomHeapType * getCustomHeap (void) {
  TheCustomHeapType * heap;
  heap = (TheCustomHeapType *) TlsGetValue (LocalTLABIndex);
  if (heap == NULL)  {
    heap = initializeCustomHeap();
  }
  return heap;
}

#ifndef HOARD_PRE_ACTION
#define HOARD_PRE_ACTION
#endif

#ifndef HOARD_POST_ACTION
#define HOARD_POST_ACTION
#endif

#ifndef CUSTOM_DLLNAME
#define CUSTOM_DLLNAME DllMain
#endif


//
// Intercept thread creation and destruction to flush the TLABs.
//

extern "C" {

  BOOL WINAPI CUSTOM_DLLNAME (HANDLE hinstDLL, DWORD fdwReason, LPVOID lpreserved)
  {
    static int np = HL::CPUInfo::computeNumProcessors();

    switch (fdwReason) {
      
    case DLL_PROCESS_ATTACH:
      {
	LocalTLABIndex = TlsAlloc();
	if (LocalTLABIndex == TLS_OUT_OF_INDEXES) {
	  // Not sure what to do here!
	}
	HOARD_PRE_ACTION;
	getCustomHeap();
      }
      break;
      
    case DLL_THREAD_ATTACH:
      if (np == 1) {
	// We have exactly one processor - just assign the thread to
	// heap 0.
	getMainHoardHeap()->chooseZero();
      } else {
	getMainHoardHeap()->findUnusedHeap();
      }
      getCustomHeap();
      break;
      
    case DLL_THREAD_DETACH:
      {
	// Dump the memory from the TLAB.
	getCustomHeap()->clear();
	
	TheCustomHeapType *heap
	  = (TheCustomHeapType *) TlsGetValue(LocalTLABIndex);
	
	if (np != 1) {
	  // If we're on a multiprocessor box, relinquish the heap
	  // assigned to this thread.
	  getMainHoardHeap()->releaseHeap();
	}
	
	if (heap != 0) {
	  TlsSetValue (LocalTLABIndex, 0);
	}
      }
      break;
      
    case DLL_PROCESS_DETACH:
      HOARD_POST_ACTION;
      break;
      
    default:
      return TRUE;
    }

    return TRUE;
  }
}
