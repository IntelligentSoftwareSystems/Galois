// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Runtime
//   B|S :: RunBuffered -> WaitForCommit -> RunCommit
//           -> WaitForSerial -> RunSerial -> WaitForBuffered
//

#include "config.h"

#ifdef DMP_ENABLE_MODEL_B_S

#include "dmp-internal.h"
#include "dmp-internal-wb.h"

//-----------------------------------------------------------------------
// API
//-----------------------------------------------------------------------

void DMP_initRuntime() {
}

void DMP_initRuntimeThread() {
  DMPwbInit();
}

void DMP_commitBufferedWrites() {
  DMPwbCommit();
}

//-----------------------------------------------------------------------
// Loads/Stores
//-----------------------------------------------------------------------

//
// Contained accesses
//

void DMPloadBufferedContained(void* addr, size_t size, void* outbuffer) {
  if (likely(DMPMAP->state == RunBuffered)) {
    DMPwbLoadContained(addr, size, outbuffer);
  } else {
    memcpy(outbuffer, addr, size);
  }
}

void DMPstoreBufferedContained(void* addr, size_t size, void* inbuffer) {
  if (likely(DMPMAP->state == RunBuffered)) {
    DMPwbStoreContained(addr, size, inbuffer);
  } else {
    memcpy(addr, inbuffer, size);
  }
}

//
// Uncontained accesses
//

void DMPloadBufferedRange(void* addr, size_t size, void* outbuffer) {
  if (likely(DMPMAP->state == RunBuffered)) {
    DMPwbLoadRange(addr, size, outbuffer);
  } else {
    memcpy(outbuffer, addr, size);
  }
}

void DMPstoreBufferedRange(void* addr, size_t size, void* inbuffer) {
  if (likely(DMPMAP->state == RunBuffered)) {
    DMPwbStoreRange(addr, size, inbuffer);
  } else {
    memcpy(addr, inbuffer, size);
  }
}

//
// Stack-allocation removal
//

void DMPremoveBufferedRange(void* addr, size_t size) {
  DMPwbRemoveRange(addr, size);
}

//
// Type specializations
//

#define INSTANTIATE(T, TNAME, KIND, CONTAINED)\
  T DMPloadBuffered ## KIND ## TNAME(T* addr) {\
    if (likely(DMPMAP->state == RunBuffered)) {\
      return DMPwbLoadTyped<T,CONTAINED>(addr);\
    } else {\
      return *addr;\
    }\
  }\
  void DMPstoreBuffered ## KIND ## TNAME(T* addr, T value) {\
    if (likely(DMPMAP->state == RunBuffered)) {\
      DMPwbStoreTyped<T,CONTAINED>(addr, value);\
    } else {\
      *addr = value;\
    }\
  }

INSTANTIATE(uint8_t,  Int8,   Contained, true)
INSTANTIATE(uint16_t, Int16,  Contained, true)
INSTANTIATE(uint32_t, Int32,  Contained, true)
INSTANTIATE(uint64_t, Int64,  Contained, true)
INSTANTIATE(float,    Float,  Contained, true)
INSTANTIATE(double,   Double, Contained, true)
INSTANTIATE(void*,    Ptr,    Contained, true)

INSTANTIATE(uint16_t, Int16,  Range, false)
INSTANTIATE(uint32_t, Int32,  Range, false)
INSTANTIATE(uint64_t, Int64,  Range, false)
INSTANTIATE(float,    Float,  Range, false)
INSTANTIATE(double,   Double, Range, false)
INSTANTIATE(void*,    Ptr,    Range, false)

#undef INSTANTIATE

//--------------------------------------------------------------
// LibC stubs
//--------------------------------------------------------------

void DMPmemset(void* addr, int val, size_t size) {
  if (likely(DMPMAP->state == RunBuffered)) {
    DMPwbMemset(addr, val, size);
  } else {
    memset(addr, val, size);
  }
}

void DMPmemcpy(void* dst, const void* src, size_t size) {
  if (likely(DMPMAP->state == RunBuffered)) {
    DMPwbMemcpy(dst, src, size);
  } else {
    memcpy(dst, src, size);
  }
}

#endif  // DMP_ENABLE_MODEL_B_S
