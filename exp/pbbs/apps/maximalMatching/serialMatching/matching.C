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

#include "graph.h"
#include "utils.h"
using namespace std;

typedef int eindex;

int maxMatchSerial(edge* E, vindex* vertices, int* out, int m) {
  int k=0;
  for (int i = 0; i < m; i++) {
    vindex v = E[i].v;
    vindex u = E[i].u;
    if(vertices[v] < 0 && vertices[u] < 0){
      vertices[v] = u;
      vertices[u] = v;
      out[k++] = i;
    }
  }
  return k;
}

// Finds a maximal matching of the graph
// Returns cross pointers between vertices, or -1 if unmatched
pair<int*,int> maximalMatching(edgeArray EA) {
  int m = EA.nonZeros;
  int n = EA.numRows;
  cout << "n=" << n << "m=" << m << endl;

  vindex* vertices = newA(vindex,n);
  for(int i=0; i<n; i++) vertices[i] = -1;
  int* out = newA(int,m);
  int size = maxMatchSerial(EA.E, vertices, out, m);
  free(vertices);

  return pair<int*,int>(out,size);
}  

