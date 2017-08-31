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
#include <cstring>
#include "parallel.h"
#include "IO.h"
#include "graph.h"
#include "graphIO.h"
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;

// Checks for every vertex if locally maximally matched
int checkMaximalMatching(edgeArray EA, int* EI, int nMatches) {
  edge* E = EA.E;
  int m = EA.nonZeros;
  int n = EA.numRows;
  int* V = newA(int, n);
//  parallel_for (int i=0; i < n; i++) V[i] = -1;
  parallel_doall(int, i, 0, n) { V[i] = -1; } parallel_doall_end
  bool *flags = newA(bool,m);
//  parallel_for (int i=0; i < m; i++) flags[i] = 0;
  parallel_doall(int, i, 0, m) { flags[i] = 0; } parallel_doall_end

//  parallel_for (int i=0; i < nMatches; i++) {
  parallel_doall(int, i, 0, nMatches)  {
    int idx = EI[i];
    V[E[idx].u] = V[E[idx].v] = idx;
    flags[idx] = 1;
  } parallel_doall_end

  for (int i=0; i < m; i++) {
      int u = E[i].u;
      int v = E[i].v;
    if (flags[i]) {
      if (V[u] != i) {
	cout << "maximalMatchingCheck: edges share vertex " << u << endl;
	return 1;
      }
      if (V[v] != i) {
	cout << "maximalMatchingCheck: edges share vertex " << v << endl;
	return 1;
      }
    } else {

      if (u != v && V[u] == -1 && V[v] == -1) {
	cout << "maximalMatchingCheck: neither endpoint matched for edge " << i << endl;
	return 1;
      }
    }
  }

  free(V); free(flags);
  return 0;
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"<inFile> <outfile>");
  pair<char*,char*> fnames = P.IOFileNames();
  edgeArray E = readEdgeArrayFromFile(fnames.first);
  _seq<int> Out = readIntArrayFromFile(fnames.second);
  return checkMaximalMatching(E, Out.A, Out.n);
}
