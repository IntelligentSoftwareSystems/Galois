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

// Checks if valid maximal independent set
int checkMaximalIndependentSet(graph G, int* Flags) {
  int n     = G.n;
  vertex* V = G.V;
  for (int i = 0; i < n; i++) {
    int nflag;
    for (int j = 0; j < V[i].degree; j++) {
      vindex ngh = V[i].Neighbors[j];
      if (Flags[ngh] == 1)
        if (Flags[i] == 1) {
          cout << "checkMaximalIndependentSet: bad edge " << i << "," << ngh
               << endl;
          return 1;
        } else
          nflag = 1;
    }
    if ((Flags[i] != 1) && (nflag != 1)) {
      cout << "checkMaximalIndependentSet: bad vertex " << i << endl;
      return 1;
    }
  }
  return 0;
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "<inFile> <outfile>");
  pair<char*, char*> fnames = P.IOFileNames();
  char* iFile               = fnames.first;
  char* oFile               = fnames.second;

  graph G       = readGraphFromFile(iFile);
  _seq<int> Out = readIntArrayFromFile(oFile);
  if (Out.n != G.n) {
    cout << "maximumMatchingCheck: output file not of right length" << endl;
    return (1);
  }

  return checkMaximalIndependentSet(G, Out.A);
}
