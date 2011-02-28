#include "Galois/Runtime/Support.h"
#include "LLVM/SmallVector.h"

#include <iostream>

void GaloisRuntime::reportStat(const char* text, unsigned long val) {
  std::cout << "STAT: " << text << " " << val << "\n";
}

void GaloisRuntime::reportStat(const char* text, unsigned int val) {
  std::cout << "STAT: " << text << " " << val << "\n";
}

void GaloisRuntime::reportStat(const char* text, double val) {
  std::cout << "STAT: " << text << " " << val << "\n";
}

//Report Warnings
void GaloisRuntime::reportWarning(const char* text) {
  std::cerr << "WARNING: " << text << "\n";
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
