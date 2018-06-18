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
#include "gettime.h"
#include "utils.h"
#include "graph.h"
#include "parallel.h"
#include "IO.h"
#include "graphIO.h"
#include "parseCommandLine.h"
#include "matching.h"
using namespace std;
using namespace benchIO;

void timeMatching(edgeArray EA, int rounds, char* outFile) {
  int m = EA.nonZeros;
  int n = EA.numRows;
  pair<int*, int> EdgeIds(NULL, 0);
  edge* E = newA(edge, m);
  for (int i = 0; i < rounds; i++) {
    if (EdgeIds.first != NULL)
      free(EdgeIds.first);
    //    parallel_for (int i=0; i < m; i++) E[i] = EA.E[i];
    parallel_doall(int, i, 0, m) { E[i] = EA.E[i]; }
    parallel_doall_end startTime();
    EdgeIds = maximalMatching(edgeArray(E, n, n, m));
    nextTimeN();
  }
  cout << endl;
  if (outFile != NULL)
    writeIntArrayToFile(EdgeIds.first, EdgeIds.second, outFile);
  free(EdgeIds.first);
  free(E);
  EA.del();
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "[-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile  = P.getArgument(0);
  char* oFile  = P.getOptionValue("-o");
  int rounds   = P.getOptionIntValue("-r", 1);
  edgeArray EA = readEdgeArrayFromFile(iFile);
  timeMatching(EA, rounds, oFile);
}
