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

#ifndef _GRAPH_INCLUDED
#define _GRAPH_INCLUDED

#include <iostream>
#include <algorithm>
#include "utils.h"
typedef int vindex;

// **************************************************************
//    SPARSE ROW MAJOR REPRESENTATION
// **************************************************************

template <class ETYPE>
struct sparseRowMajor {
  int numRows;
  int numCols;
  int nonZeros;
  int* Starts;
  int* ColIds;
  ETYPE* Values;
  void del() {free(Starts); free(ColIds); if (Values != NULL) free(Values);}
  sparseRowMajor(int n, int m, int nz, int* S, int* C, ETYPE* V) :
    numRows(n), numCols(m), nonZeros(nz), 
    Starts(S), ColIds(C), Values(V) {}
};

typedef sparseRowMajor<double> sparseRowMajorD;

// **************************************************************
//    EDGE ARRAY REPRESENTATION
// **************************************************************

struct edge {
  vindex u;
  vindex v;
  edge(vindex f, vindex s) : u(f), v(s) {}
};

struct edgeArray {
  edge* E;
  int numRows;
  int numCols;
  int nonZeros;
  void del() {free(E);}
  edgeArray(edge *EE, int r, int c, int nz) :
    E(EE), numRows(r), numCols(c), nonZeros(nz) {}
  edgeArray() {}
};

// **************************************************************
//    WEIGHED EDGE ARRAY
// **************************************************************

struct wghEdge {
  vindex u, v;
  double weight;
  wghEdge() {}
  wghEdge(vindex _u, vindex _v, double w) : u(_u), v(_v), weight(w) {}
};

struct wghEdgeArray {
  wghEdge *E;
  int n; int m;
  wghEdgeArray(wghEdge* EE, int nn, int mm) : E(EE), n(nn), m(mm) {}
  void del() { free(E);}
};

// **************************************************************
//    ADJACENCY ARRAY REPRESENTATION
// **************************************************************

struct vertex {
  vindex* Neighbors;
  int degree;
  void del() {free(Neighbors);}
  vertex(vindex* N, int d) : Neighbors(N), degree(d) {}
};

struct graph {
  vertex *V;
  int n;
  int m;
  vindex* allocatedInplace;
  graph(vertex* VV, int nn, int mm) 
    : V(VV), n(nn), m(mm), allocatedInplace(NULL) {}
  graph(vertex* VV, int nn, int mm, vindex* ai) 
    : V(VV), n(nn), m(mm), allocatedInplace(ai) {}
  graph copy() {
    vertex* VN = newA(vertex,n);
    vindex* Edges = newA(vindex,m);
    vindex k = 0;
    for (int i=0; i < n; i++) {
      VN[i] = V[i];
      VN[i].Neighbors = Edges + k;
      for (int j =0; j < V[i].degree; j++) 
	Edges[k++] = V[i].Neighbors[j];
    }
    return graph(VN, n, m, Edges);
  } 
  void del() {
    if (allocatedInplace == NULL) 
      for (int i=0; i < n; i++) V[i].del();
    else free(allocatedInplace);
    free(V);
  }
};

#endif // _GRAPH_INCLUDED
