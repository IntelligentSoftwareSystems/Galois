#ifndef TYPES_H
#define TYPES_H

#include "galois/Galois.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/SimpleLock.h"

// We provide two types of 'support': frequency and domain support.
// Frequency is used for counting, e.g. motif counting.
// Domain support, a.k.a, the minimum image-based support, is used for FSM. It has the anti-monotonic property.
typedef float MatType;
typedef unsigned Frequency;
typedef std::vector<std::vector<MatType> > Matrix;
typedef galois::GAccumulator<unsigned> UintAccu;
typedef galois::GAccumulator<unsigned long> UlongAccu;
typedef std::unordered_map<unsigned, unsigned> UintMap;
typedef galois::substrate::PerThreadStorage<UintMap> LocalUintMap;

#endif
