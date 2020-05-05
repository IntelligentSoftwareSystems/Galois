#pragma once
#include "pangolin/types.h"
#include "pangolin/edge_embedding.h"
#include "pangolin/quick_pattern.h"
#include "pangolin/canonical_graph.h"

typedef QuickPattern<EdgeInducedEmbedding<StructuralElement>, StructuralElement>
    StrQPattern; // structural quick pattern
typedef CanonicalGraph<EdgeInducedEmbedding<StructuralElement>,
                       StructuralElement>
    StrCPattern; // structural canonical pattern
typedef std::unordered_map<StrQPattern, Frequency>
    StrQpMapFreq; // mapping structural quick pattern to its frequency
typedef std::unordered_map<StrCPattern, Frequency>
    StrCgMapFreq; // mapping structural canonical pattern to its frequency
typedef galois::substrate::PerThreadStorage<StrQpMapFreq> LocalStrQpMapFreq;
typedef galois::substrate::PerThreadStorage<StrCgMapFreq> LocalStrCgMapFreq;
/*
class Status {
protected:
    std::vector<uint8_t> visited;
public:
    Status() {}
    ~Status() {}
    void init(unsigned size) {
        visited.resize(size);
        reset();
    }
    void reset() {
        std::fill(visited.begin(), visited.end(), 0);
    }
    void set(VertexId pos, uint8_t value) { visited[pos] = value; }
    uint8_t get(VertexId pos) { return visited[pos]; }
};
typedef galois::substrate::PerThreadStorage<Status> StatusMT; // multi-threaded
*/
