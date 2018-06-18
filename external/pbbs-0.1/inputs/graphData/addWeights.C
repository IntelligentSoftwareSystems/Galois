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

// Adds a random double precision weight to each edge

#include <math.h>
#include "IO.h"
#include "graph.h"
#include "graphIO.h"
#include "parseCommandLine.h"
#include "dataGen.h"
#include "parallel.h"
using namespace benchIO;
using namespace dataGen;
using namespace std;

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "<inFile> <outFile>");
  pair<char*, char*> fnames = P.IOFileNames();
  char* iFile               = fnames.first;
  char* oFile               = fnames.second;

  edgeArray In = readEdgeArrayFromFile(iFile);
  int m        = In.nonZeros;
  int n        = max(In.numCols, In.numRows);
  edge* E      = In.E;
  wghEdge* WE  = newA(wghEdge, m);
  //  parallel_for(int i=0; i < m; i++) {
  parallel_doall(int, i, 0, m) {
    WE[i] = wghEdge(E[i].u, E[i].v, dataGen::hash<double>(i));
  }
  parallel_doall_end return writeWghEdgeArrayToFile(wghEdgeArray(WE, n, m),
                                                    oFile);
}
