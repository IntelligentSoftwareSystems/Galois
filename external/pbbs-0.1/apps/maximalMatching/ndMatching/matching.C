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
#include "parallel.h"
#include "sequence.h"
#include "graph.h"
#include "utils.h"
using namespace std;

void maxMatchNonDeterministic(edge* E, bool* matched, vindex* vertices, int m) {
  //  parallel_for (int i = 0; i < m; i++) {
  parallel_doall(int, i, 0, m) {
    vindex v = E[i].v;
    vindex u = E[i].u;
    int j    = 0;
    if (u != v) {
      while (1) {
        if (matched[v] || matched[u])
          break;
        if (utils::CAS(&vertices[v], -1, -2)) {
          if (utils::CAS(&vertices[u], -1, -2)) {
            matched[v] = matched[u] = 1;
            vertices[u]             = i;
            break;
          } else
            vertices[v] = -1;
        }
      }
    }
  }
  parallel_doall_end
}

void initVertices(bool* matched, vindex* vertices, int n) {
  //  parallel_for(int i=0;i<n;i++) {
  parallel_doall(int, i, 0, n) {
    vertices[i] = -1;
    matched[i]  = 0;
  }
  parallel_doall_end
}

struct nonNegative {
  bool operator()(int i) { return i >= 0; }
};

// Finds a maximal matching of the graph
// Returns cross pointers between vertices, or -1 if unmatched
pair<int*, int> maximalMatching(edgeArray EA) {
  int m = EA.nonZeros;
  int n = EA.numRows;

  vindex* R     = newA(vindex, n);
  bool* matched = newA(bool, n);
  initVertices(matched, R, n);

  maxMatchNonDeterministic(EA.E, matched, R, m);

  int* Out     = newA(int, n);
  int nMatches = sequence::filter(R, Out, n, nonNegative());
  free(matched);
  free(R);

  cout << "number of matches = " << nMatches << endl;
  return pair<int*, int>(Out, nMatches);
}
