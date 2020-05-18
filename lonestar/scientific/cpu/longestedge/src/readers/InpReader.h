#ifndef INP_READER_H
#define INP_READER_H

#include "../model/Graph.h"

#include <string>

//!
//! \brief inpRead Function that reads a mesh in INP format generates the graph,
//! and computes the four corners of the mesh. \param filename Path of the INP
//! file. \param graph Galois hypergraph representing the mesh. \param config
//! Config object. The four corners E, W, N, S are populated in the function.
//! version2D is used for the edge length
//!

void inpRead(const std::string& filename, Graph& graph, double N, double S,
             double E, double W, bool version2D);

#endif // INP_READER_H
