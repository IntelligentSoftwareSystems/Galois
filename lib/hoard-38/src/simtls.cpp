static pthread_key_t theHeapKey;
static pthread_once_t key_once = PTHREAD_ONCE_INIT;

// Called when the thread goes away.  This function clears out the
// TLAB and then reclaims the memory allocated to hold it.

static void deleteThatHeap (void * p) {
  TheCustomHeapType * heap = (TheCustomHeapType *) p;
  heap->clear();
  getMainHoardHeap()->free ((void *) heap);

  // Relinquish the assigned heap.
  getMainHoardHeap()->releaseHeap();
}

static void make_heap_key (void)
{
  if (pthread_key_create (&theHeapKey, deleteThatHeap) != 0) {
    // This should never happen.
  }
}

static void initTSD (void) __attribute__((constructor));

static void initTSD (void) {
  static bool initializedTSD = false;
  if (!initializedTSD) {
    // Ensure that the key is initialized -- once.
    pthread_once (&key_once, make_heap_key);
    initializedTSD = true;
  }
}

static TheCustomHeapType * initializeCustomHeap (void)
{
  assert (pthread_getspecific (theHeapKey) == NULL);

  // Allocate a per-thread heap.
  TheCustomHeapType * heap;
  size_t sz = sizeof(TheCustomHeapType) + sizeof(double);
  void * mh = getMainHoardHeap()->malloc(sz);
  heap = new ((char *) mh) TheCustomHeapType (getMainHoardHeap());

  // Store it in the appropriate thread-local area.
  int r = pthread_setspecific (theHeapKey, (void *) heap);
  r = r;
  assert (!r);

  return heap;
}

inline TheCustomHeapType * getCustomHeap (void) {
  TheCustomHeapType * heap;
  initTSD();
  heap = (TheCustomHeapType *) pthread_getspecific (theHeapKey);
  if (heap == NULL)  {
    heap = initializeCustomHeap();
  }
  return heap;
}
