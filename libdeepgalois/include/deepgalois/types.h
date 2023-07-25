#ifndef _GNN_TYPES_H_
#define _GNN_TYPES_H_
#include <set>
#include <vector>
#include <stdint.h>
#include <cstddef>

// TODO namespace

#ifdef CNN_USE_DOUBLE
typedef double float_t;
typedef double feature_t;
#else
typedef float float_t;
typedef float feature_t; // feature type
#endif
typedef std::vector<float_t> vec_t; // feature vector (1D)
typedef std::vector<vec_t>
    tensor_t; // feature vectors (2D): num_samples x feature_dim
typedef std::vector<feature_t> FV; // feature vector
typedef std::vector<FV> FV2D;      // feature vectors: num_samples x feature_dim
typedef float acc_t;               // Accuracy type
typedef uint8_t label_t; // label is for classification (supervised learning)
typedef uint8_t mask_t;  // mask is used to indicate different uses of labels:
                         // train, val, test
typedef uint32_t VertexID;
typedef uint64_t EdgeID;
typedef std::vector<VertexID> VertexList;
typedef std::set<VertexID> VertexSet;
typedef std::vector<size_t> dims_t; // dimentions type

typedef uint32_t index_t; // index type
typedef float_t edata_t;  // edge data type
typedef float_t vdata_t;  // vertex data type
typedef float_t* emb_t;   // embedding (feature vector) type

enum class net_phase { train, test };

#define CHUNK_SIZE 256
#define TB_SIZE 256
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_NUM_CLASSES 128
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define USE_CUSPARSE

namespace deepgalois {
// TODO only being used by graph conv layer at the moment so extern works,
// but this design is bad and needs to be revisited

//! Set this to let sync struct know where to get data from
extern float_t* _dataToSync;
//! Set this to let sync struct know the size of the vector to use during
//! sync
extern long unsigned _syncVectorSize;
} // namespace deepgalois

#endif
