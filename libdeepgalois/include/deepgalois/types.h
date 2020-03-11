#ifndef TYPES_H
#define TYPES_H
#include <vector>
#include <stdint.h>

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
typedef short label_t;  // label is for classification (supervised learning)
typedef uint8_t mask_t; // mask is used to indicate different uses of labels:
                        // train, val, test
#define CHUNK_SIZE 256
#define TB_SIZE 256
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_NUM_CLASSES 64
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define USE_CUSPARSE
#endif
