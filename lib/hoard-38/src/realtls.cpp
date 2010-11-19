// Thread-specific buffers and pointers to hold the TLAB.

static __thread double tlabBuffer[sizeof(TheCustomHeapType) / sizeof(double) + 1];
static __thread TheCustomHeapType * theTLAB = NULL;

// Initialize the TLAB (must only be called once).

static TheCustomHeapType * initializeCustomHeap (void) {
  new ((char *) &tlabBuffer) TheCustomHeapType (getMainHoardHeap());
  return (theTLAB = (TheCustomHeapType *) &tlabBuffer);
}

// Get the TLAB.

inline TheCustomHeapType * getCustomHeap (void) {
  // The pointer to the TLAB itself.
  theTLAB = (theTLAB ? theTLAB : initializeCustomHeap());
  return theTLAB;
}
