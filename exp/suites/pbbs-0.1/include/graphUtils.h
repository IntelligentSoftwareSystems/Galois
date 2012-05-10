// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _GRAPH_UTILS_INCLUDED
#define _GRAPH_UTILS_INCLUDED

#include "graph.h"

// Most of the routines in here convert between different types of
// graph representations

wghEdgeArray addRandWeights(edgeArray G);

edgeArray edgesFromSparse(sparseRowMajorD M);

edgeArray edgesFromGraph(graph G);

// removes duplicate edges
edgeArray remDuplicates(edgeArray A);

// adds edges so all edges appear in both directions
edgeArray makeSymmetric(edgeArray A); 

graph graphFromEdges(edgeArray EA, bool makeSym);

sparseRowMajorD sparseFromGraph(graph G);

// if I is NULL then it randomly reorders
graph graphReorder(graph Gr, int* I);

int graphCheckConsistency(graph Gr);

#endif // _GRAPH_UTILS_INCLUDED
