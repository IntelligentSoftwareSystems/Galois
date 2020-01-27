#ifndef TYPES_H
#define TYPES_H
#include <vector>

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

#endif
