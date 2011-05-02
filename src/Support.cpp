#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Runtime/Support.h"
#include "LLVM/SmallVector.h"
#include <iostream>

static GaloisRuntime::SimpleLock<int, true> lock;

void GaloisRuntime::reportStat(const char* text, unsigned long val) {
  lock.lock();
  std::cout << "STAT: " << text << " " << val << "\n";
  lock.unlock();
}

void GaloisRuntime::reportStat(const char* text, unsigned int val) {
  lock.lock();
  std::cout << "STAT: " << text << " " << val << "\n";
  lock.unlock();
}

void GaloisRuntime::reportStat(const char* text, double val) {
  lock.lock();
  std::cout << "STAT: " << text << " " << val << "\n";
  lock.unlock();
}

//Report Warnings
void GaloisRuntime::reportWarning(const char* text) {
  lock.lock();
  std::cerr << "WARNING: " << text << "\n";
  lock.unlock();
}

void GaloisRuntime::reportWarning(const char* text, unsigned int val) {
  lock.lock();
  std::cerr << "WARNING: " << text << " " << val << "\n";
  lock.unlock();
}

void GaloisRuntime::reportWarning(const char* text, unsigned long val) {
  lock.lock();
  std::cerr << "WARNING: " << text << " " << val << "\n";
  lock.unlock();
}

void GaloisRuntime::reportWarning(const char* text, const char* val) {
  lock.lock();
  std::cerr << "WARNING: " << text << " " << val << "\n";
  lock.unlock();
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
