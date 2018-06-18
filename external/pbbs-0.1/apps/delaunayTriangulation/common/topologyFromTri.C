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

#include <iostream>
#include <algorithm>
#include "parallel.h"
#include "gettime.h"
#include "geometry.h"
#include "topology.h"
#include "deterministicHash.h"
using namespace std;

typedef pair<int, int> pairInt;
typedef pair<pairInt, tri*> edge;

// Hash table to store skinny triangles
struct hashEdges {
  typedef pairInt kType;
  typedef edge* eType;
  eType empty() { return NULL; }
  kType getKey(eType v) { return v->first; }
  unsigned int hash(kType s) {
    return utils::hash(s.first) + 3 * (utils::hash(s.second));
  }
  int cmp(kType s1, kType s2) {
    return ((s1.first > s2.first)
                ? 1
                : (s1.first < s2.first)
                      ? -1
                      : (s1.second > s2.second)
                            ? 1
                            : (s1.second < s2.second) ? -1 : 0);
  }
  bool replaceQ(eType s, eType s2) { return 0; }
};

typedef Table<hashEdges> EdgeTable;
EdgeTable makeEdgeTable(int m) { return EdgeTable(m, hashEdges()); }

void topologyFromTriangles(triangles<point2d> Tri, vertex** vr, tri** tr) {
  int n      = Tri.numPoints;
  point2d* P = Tri.P;

  int m       = Tri.numTriangles;
  triangle* T = Tri.T;

  if (*vr == NULL)
    *vr = newA(vertex, n);
  vertex* v = *vr;
  //  parallel_for (int i=0; i < n; i++)
  parallel_doall(int, i, 0, n) { v[i] = vertex(P[i], i); }
  parallel_doall_end

      if (*tr == NULL)* tr = newA(tri, m);
  tri* Triangs             = *tr;
  edge* E                  = newA(edge, m * 3);
  EdgeTable ET             = makeEdgeTable(m * 6);
  //  parallel_for (int i=0; i < m; i++)
  parallel_doall(int, i, 0, m) {
    for (int j = 0; j < 3; j++) {
      E[i * 3 + j] = edge(pairInt(T[i].C[j], T[i].C[(j + 1) % 3]), &Triangs[i]);
      ET.insert(&E[i * 3 + j]);
      Triangs[i].vtx[(j + 2) % 3] = &v[T[i].C[j]];
    }
  }
  parallel_doall_end

  //  parallel_for (int i=0; i < m; i++) {
  parallel_doall(int, i, 0, m) {
    Triangs[i].id          = i;
    Triangs[i].initialized = 1;
    Triangs[i].bad         = 0;
    for (int j = 0; j < 3; j++) {
      pairInt key = pairInt(T[i].C[(j + 1) % 3], T[i].C[j]);
      edge* Ed    = ET.find(key);
      if (Ed != NULL)
        Triangs[i].ngh[j] = Ed->second;
      else {
        Triangs[i].ngh[j] = NULL;
        // Triangs[i].vtx[j]->boundary = 1;
        // Triangs[i].vtx[(j+2)%3]->boundary = 1;
      }
    }
  }
  parallel_doall_end

      ET.del();
  free(E);
}
