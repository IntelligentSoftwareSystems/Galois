#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Runtime/Support.h"
#include "LLVM/SmallVector.h"
#include <iostream>

static GaloisRuntime::SimpleLock<int, true> lock;

template<typename T>
static void genericReport(bool error, const char* text1, const char* text2, T val) {
  lock.lock();
  (error ? std::cerr : std::cout) <<
    text1 << " " << text2 << " " << val << "\n";
  lock.unlock();
}

void GaloisRuntime::reportStat(const char* text, unsigned long val) {
  genericReport(false, "STAT:", text, val);
}

void GaloisRuntime::reportStat(const char* text, unsigned int val) {
  genericReport(false, "STAT:", text, val);
}

void GaloisRuntime::reportStat(const char* text, double val) {
  genericReport(false, "STAT:", text, val);
}

//Report Warnings
void GaloisRuntime::reportWarning(const char* text) {
  genericReport(true, "WARNING:", text, "");
}

void GaloisRuntime::reportWarning(const char* text, unsigned int val) {
  genericReport(true, "WARNING:", text, val);
}

void GaloisRuntime::reportWarning(const char* text, unsigned long val) {
  genericReport(true, "WARNING:", text, val);
}

void GaloisRuntime::reportWarning(const char* text, const char* val) {
  genericReport(true, "WARNING:", text, val);
}

/// grow_pod - This is an implementation of the grow() method which only works
/// on POD-like datatypes and is out of line to reduce code duplication.
void llvm::SmallVectorBase::grow_pod(size_t MinSizeInBytes, size_t TSize) {
  size_t CurSizeBytes = size_in_bytes();
  size_t NewCapacityInBytes = 2 * capacity_in_bytes() + TSize; // Always grow.                                                                                                                      
  if (NewCapacityInBytes < MinSizeInBytes)
    NewCapacityInBytes = MinSizeInBytes;

  void *NewElts;
  if (this->isSmall()) {
    NewElts = malloc(NewCapacityInBytes);

    // Copy the elements over.  No need to run dtors on PODs.                                                                                                                                       
    memcpy(NewElts, this->BeginX, CurSizeBytes);
  } else {
    // If this wasn't grown from the inline copy, grow the allocated space.                                                                                                                         
    NewElts = realloc(this->BeginX, NewCapacityInBytes);
  }

  this->EndX = (char*)NewElts+CurSizeBytes;
  this->BeginX = NewElts;
  this->CapacityX = (char*)this->BeginX + NewCapacityInBytes;
}
