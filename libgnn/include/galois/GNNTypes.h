#pragma once
//! @file GNNTypes.h
//! Typedefs used by the Galois GNN code

#include <cstdint>
#include <cstddef>

namespace galois {
//! Floating point type to use throughout GNN compute; typedef'd so it's easier
//! to flip later
using GNNFloat = float;
//! Type of the labels for a vertex
using GNNLabel = uint8_t;
//! Type of a feature on vertices
using GNNFeature = float;
} // end namespace galois
