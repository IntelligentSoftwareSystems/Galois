#ifndef TYPES_H
#define TYPES_H
// common types
#include <map>
#include <set>
#include <queue>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#define CHUNK_SIZE 256
#define LARGE_SIZE // for large graphs such as soc-Livejournal1 and com-Orkut

typedef float Weight;
typedef uint64_t Ulong;
typedef uint32_t ValueT;
typedef uint32_t VertexId;
typedef uint64_t EdgeId;
typedef uint8_t BYTE;
#ifdef LARGE_SIZE
typedef uint64_t IndexT;
typedef uint64_t IndexTy;
#else
typedef uint32_t IndexT;
typedef uint32_t IndexTy;
#endif

typedef std::set<uint32_t> UintSet;
typedef std::vector<UintSet> UintSets;

#ifdef LARGE_SIZE
typedef std::vector<BYTE> ByteList;
typedef std::vector<uint32_t> UintList;
typedef std::vector<Ulong> UlongList;
typedef std::vector<VertexId> VertexList;
#endif
typedef std::vector<UintList> IndexLists;
typedef std::vector<ByteList> ByteLists;
typedef std::vector<VertexList> VertexLists;
typedef std::vector<bool> BoolVec;

// We provide two types of 'support': frequency and domain support.
// Frequency is used for counting, e.g. motif counting.
// Domain support, a.k.a, the minimum image-based support, is used for FSM. It has the anti-monotonic property.
typedef float MatType;
typedef unsigned Frequency;
typedef std::vector<std::vector<MatType> > Matrix;
typedef std::unordered_map<unsigned, unsigned> UintMap;
typedef std::pair<unsigned, unsigned> InitPattern;
typedef std::unordered_map<unsigned, unsigned> FreqMap;
typedef std::unordered_map<unsigned, bool> DomainMap;
typedef std::map<unsigned, std::map<unsigned, unsigned> > Map2D;

#endif
