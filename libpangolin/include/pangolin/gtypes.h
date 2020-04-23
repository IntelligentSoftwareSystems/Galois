#pragma once
// Galois supported types
#include "pangolin/types.h"
#include "galois/Bag.h"
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/SimpleLock.h"

#ifndef LARGE_SIZE
typedef galois::gstl::Vector<BYTE> ByteList;
typedef galois::gstl::Vector<unsigned> UintList;
typedef galois::gstl::Vector<Ulong> UlongList;
typedef galois::gstl::Vector<VertexId> VertexList;
#endif

typedef galois::gstl::Set<VertexId> VertexSet;
typedef galois::substrate::PerThreadStorage<UintList> Lists;
typedef galois::substrate::PerThreadStorage<unsigned> Counts;

typedef galois::GAccumulator<unsigned> UintAccu;
typedef galois::GAccumulator<uint64_t> UlongAccu;
typedef galois::substrate::PerThreadStorage<UintMap> LocalUintMap;

//typedef galois::gstl::Map<unsigned, unsigned> FreqMap;
//typedef galois::gstl::UnorderedMap<unsigned, bool> DomainMap;

// use Galois memory allocator for domain support
typedef galois::gstl::Set<int> IntSet;
typedef galois::gstl::Vector<IntSet> IntSets;
//typedef std::set<int> IntSet;
//typedef std::vector<IntSet> IntSets;

typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

