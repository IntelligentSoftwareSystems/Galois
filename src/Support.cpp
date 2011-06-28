/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/
#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Runtime/Support.h"
#include "LLVM/SmallVector.h"
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdio.h>

static GaloisRuntime::SimpleLock<int, true> lock;

void GaloisRuntime::summarizeList(const char* name, const long* b, const long* e) {
  long size = std::distance(b,e);
  long min = *std::min_element(b, e);
  long max = *std::max_element(b, e);
  double ave = std::accumulate(b, e, 0.0) / size;
 
  double acc = 0.0;
  for (const long* it = b; it != e; ++it) {
    acc += (*it - ave) * (*it - ave);
  }

  double stdev = 0.0;
  if (size > 1) {
    stdev = sqrt(acc / (size - 1));
  }

  std::ostringstream out;
  out.setf(std::ios::fixed, std::ios::floatfield);
  out.precision(1);
  out << "n: " << size;
  out << " ave: " << ave;
  out << " min: " << min;
  out << " max: " << max;
  out << " stdev: " << stdev;

  reportStat(name, out.str().c_str());
}


static void genericReport(bool error, const char* text1,
    const char* text2, const char* val) {
  lock.lock();
  FILE *out = error ? stderr : stdout;
  fprintf(out, "%s %s %s\n", text1, text2, val);
  lock.unlock();
}

void GaloisRuntime::reportStat(const char* text, unsigned long val) {
  char buf[128];
  snprintf(buf, 128, "%lu", val);
  genericReport(false, "STAT:", text, buf);
}

void GaloisRuntime::reportStat(const char* text, unsigned int val) {
  char buf[128];
  snprintf(buf, 128, "%u", val);
  genericReport(false, "STAT:", text, buf);
}

void GaloisRuntime::reportStat(const char* text, double val) {
  char buf[128];
  snprintf(buf, 128, "%f", val);
  genericReport(false, "STAT:", text, buf);
}

void GaloisRuntime::reportStat(const char* text, const char* val) {
  genericReport(false, "STAT:", text, val);
}

//Report Warnings
void GaloisRuntime::reportWarning(const char* text) {
  genericReport(true, "WARNING:", text, "");
}

void GaloisRuntime::reportWarning(const char* text, unsigned int val) {
  char buf[128];
  snprintf(buf, 128, "%u", val);
  genericReport(true, "WARNING:", text, buf);
}

void GaloisRuntime::reportWarning(const char* text, unsigned long val) {
  char buf[128];
  snprintf(buf, 128, "%lu", val);
  genericReport(true, "WARNING:", text, buf);
}

void GaloisRuntime::reportWarning(const char* text, const char* val) {
  genericReport(true, "WARNING:", text, val);
}

//Report Info
void GaloisRuntime::reportInfo(const char* text) {
  genericReport(false, "INFO:", text, "");
}

void GaloisRuntime::reportInfo(const char* text, unsigned int val) {
  char buf[128];
  snprintf(buf, 128, "%u", val);
  genericReport(false, "INFO:", text, buf);
}

void GaloisRuntime::reportInfo(const char* text, unsigned long val) {
  char buf[128];
  snprintf(buf, 128, "%lu", val);
  genericReport(false, "INFO:", text, buf);
}

void GaloisRuntime::reportInfo(const char* text, const char* val) {
  genericReport(false, "INFO:", text, val);
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
