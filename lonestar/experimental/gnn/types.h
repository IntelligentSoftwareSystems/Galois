#ifndef TYPES_H
#define TYPES_H
#include <vector>
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

#ifdef CNN_USE_DOUBLE
typedef double float_t;
typedef double feature_t;
#else
typedef float float_t;
typedef float feature_t; // feature type
#endif
typedef std::vector<float_t> vec_t; // feature vector (1D)
typedef std::vector<vec_t> tensor_t; // feature vectors (2D): num_samples x feature_dim
typedef std::vector<feature_t> FV; // feature vector
typedef std::vector<FV> FV2D; // feature vectors: num_samples x feature_dim
typedef std::vector<FV2D> FV3D; // matrices 
typedef float acc_t; // Accuracy type
typedef short label_t; // label is for classification (supervised learning)
typedef unsigned short mask_t; // mask is used to indicate different uses of labels: train, val, test
typedef std::vector<label_t> LabelList; // label list to store label for each vertex
typedef std::vector<mask_t> MaskList; // mask list to store mask for each vertex

#ifdef EDGE_LABEL
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#else
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
#endif

typedef Graph::GraphNode GNode;

#endif
