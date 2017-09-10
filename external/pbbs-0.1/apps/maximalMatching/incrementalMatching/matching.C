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
#include "limits.h"
#include "sequence.h"
#include "parallel.h"
#include "graph.h"
#include "utils.h"
#include "speculative_for.h"
using namespace std;

struct matchStep {
  edge *E;  
  int *R;  
  bool *matched;
  matchStep(edge* _E, int* _R, bool* m) : E(_E), R(_R), matched(m) {}

  bool reserve(int i) {
    int u = E[i].u;
    int v = E[i].v;
    if (matched[u] || matched[v] || (u == v)) return 0;
    reserveLoc(R[u], i);
    reserveLoc(R[v], i);
    return 1;
  }

  bool commit(int i) {
    int u = E[i].u;
    int v = E[i].v;
    if (R[v] == i) {
      R[v] = INT_MAX;
      if (R[u] == i) {
	matched[u] = matched[v] = 1;
	return 1;
      } 
    } else if (R[u] == i) R[u] = INT_MAX;
    return 0;
  }
};

struct notMax { bool operator() (int i) {return i < INT_MAX;}};

pair<int*,int> maximalMatching(edgeArray G) {
  int n = G.numRows;
  int m = G.nonZeros;
  int *R = newArray(n, INT_MAX);
  bool *matched = newArray(n, (bool) 0);
  matchStep mStep(G.E, R, matched);
  speculative_for(mStep, 0, m, 50, 0);
  _seq<int> matchingIdx = sequence::filter(R, n, notMax());
  free(R); free(matched);
  cout << "number of matches = " << matchingIdx.n << endl;
  return pair<int*,int>(matchingIdx.A, matchingIdx.n);
}  
