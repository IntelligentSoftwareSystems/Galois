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
#include "MIS.h"
using namespace std;
using namespace benchIO;

static bool CheckResult;

int checkMaximalIndependentSet(graph G, char* Flags) {
  int n = G.n;
  vertex* V = G.V;
  for (int i=0; i < n; i++) {
    int nflag;
    for (int j=0; j < V[i].degree; j++) {
      vindex ngh = V[i].Neighbors[j];
      if (Flags[ngh] == 1)
	if (Flags[i] == 1) {
	  cout << "checkMaximalIndependentSet: bad edge " 
	       << i << "," << ngh << endl;
	  return 1;
	} else nflag = 1;
    }
    if ((Flags[i] != 1) && (nflag != 1)) {
      cout << "checkMaximalIndependentSet: bad vertex " << i << endl;
      return 1;
    }
  }
  return 0;
}

void timeMIS(graph G, int rounds, char* outFile) {
  char* flags = NULL;
  //maximalIndependentSet(G);
  for (int i=0; i < rounds; i++) {
    if (flags)
      free(flags);
    startTime();
    flags = maximalIndependentSet(G);
    nextTimeN();
  }
  cout << endl;

  int matched = 0;
  for (int i=0; i < G.n; i++) if (flags[i] == 1) matched++;
  cout << "matched = " << matched << "\n";

  if (outFile != NULL) {
    int* F = newA(int, G.n);
    for (int i=0; i < G.n; i++) F[i] = flags[i];
    writeIntArrayToFile(F, G.n, outFile);
    free(F);
  }

  if (CheckResult) { if (!checkMaximalIndependentSet(G, flags)) { cout << "result ok\n"; } else { abort(); } }

  free(flags);
  G.del();
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "[-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  int rounds = P.getOptionIntValue("-r",1);
  CheckResult = P.getOption("-c");

  graph G = readGraphFromFile(iFile);
  timeMIS(G, rounds, oFile);
}
