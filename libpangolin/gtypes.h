#pragma once
#include "types.h"
#include "galois/Bag.h"
#include "galois/Galois.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/SimpleLock.h"

#ifdef LARGE_SIZE
typedef std::vector<BYTE> ByteList;
typedef std::vector<unsigned> UintList;
typedef std::vector<Ulong> UlongList;
typedef std::vector<VertexId> VertexList;
#else
typedef galois::gstl::Vector<BYTE> ByteList;
typedef galois::gstl::Vector<unsigned> UintList;
typedef galois::gstl::Vector<Ulong> UlongList;
typedef galois::gstl::Vector<VertexId> VertexList;
#endif
typedef std::vector<UintList> IndexLists;
typedef std::vector<ByteList> ByteLists;
typedef std::vector<VertexList> VertexLists;

typedef galois::gstl::Set<VertexId> VertexSet;
typedef galois::substrate::PerThreadStorage<UintList> Lists;
typedef galois::substrate::PerThreadStorage<unsigned> Counts;

typedef galois::GAccumulator<unsigned> UintAccu;
typedef galois::GAccumulator<unsigned long> UlongAccu;
typedef galois::substrate::PerThreadStorage<UintMap> LocalUintMap;

//typedef galois::gstl::Map<unsigned, unsigned> FreqMap;
//typedef galois::gstl::UnorderedMap<unsigned, bool> DomainMap;

#ifdef USE_GSTL
typedef galois::gstl::Set<int> IntSet;
//typedef galois::gstl::Vector<bool> BoolVec;
typedef galois::gstl::Vector<IntSet> IntSets;
#else
typedef std::set<int> IntSet;
//typedef std::vector<bool> BoolVec;
typedef std::vector<IntSet> IntSets;
#endif

