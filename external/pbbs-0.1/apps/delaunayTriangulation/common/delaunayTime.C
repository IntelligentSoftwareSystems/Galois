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
#include "geometry.h"
#include "geometryIO.h"
#include "parseCommandLine.h"
#include "parallel.h"
#include "delaunay.h"
using namespace std;
using namespace benchIO;

// *************************************************************
//  TIMING
// *************************************************************

bool CheckResult;

void timeDelaunay(point2d* pts, int n, int rounds, char* outFile) {
  triangles<point2d> R;
  for (int i = 0; i < rounds; i++) {
    if (i != 0)
      R.del();
    startTime();
    R = delaunay(pts, n);
    nextTimeN();
  }
  cout << endl;

  if (outFile != NULL)
    writeTrianglesToFile(R, outFile);
  R.del();
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "[-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  int rounds  = P.getOptionIntValue("-r", 1);
  CheckResult = P.getOption("-c");

  _seq<point2d> PIn = readPointsFromFile<point2d>(iFile);
  // std::random_shuffle(PIn.A, PIn.A+PIn.n);
  timeDelaunay(PIn.A, PIn.n, rounds, oFile);
}
