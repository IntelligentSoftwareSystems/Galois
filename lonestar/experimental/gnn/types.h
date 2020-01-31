#ifndef TYPES_H
#define TYPES_H
#include <vector>
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

#ifdef CNN_USE_DOUBLE
typedef double float_t;
#else
typedef float float_t;
#endif
typedef std::vector<float_t> vec_t;
typedef std::vector<vec_t> tensor_t;
typedef unsigned IndexT;
typedef float ValueT;
typedef unsigned VertexID;
typedef unsigned short MaskT;
typedef float AccT; // Accuracy type
typedef float FeatureT; // feature type
typedef std::vector<FeatureT> FV; // feature vector
typedef std::vector<FV> FV2D; // feature vectors
typedef std::vector<FV2D> FV3D; // matrices 
typedef short LabelT; // label is for classification (supervised learning)
typedef std::vector<LabelT> LabelList; // label list
typedef std::vector<MaskT> MaskList; // label list

#ifdef EDGE_LABEL
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#else
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#endif

typedef Graph::GraphNode GNode;

#endif
