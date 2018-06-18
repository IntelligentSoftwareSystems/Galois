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
#include "parallel.h"
#include "geometryIO.h"
#include "parseCommandLine.h"
#include "ray.h"

using namespace std;
using namespace benchIO;

void timeRayCast(triangles<pointT> T, ray<pointT>* rays, int nRays, int rounds,
                 char* outFile) {
  int* m = NULL;
  for (int i = 0; i < rounds; i++) {
    if (m != NULL)
      free(m);
    startTime();
    m = rayCast(T, rays, nRays);
    nextTimeN();
  }
  cout << endl;
  if (outFile != NULL)
    writeIntArrayToFile(m, nRays, outFile);
  free(m);
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv,
                "[-o <outFile>] [-r <rounds>] <triangleFile> <rayFile>");
  pair<char*, char*> fnames = P.IOFileNames();
  char* triFile             = fnames.first;
  char* rayFile             = fnames.second;
  char* oFile               = P.getOptionValue("-o");
  int rounds                = P.getOptionIntValue("-r", 1);

  // the 1 argument means that the vertices are labeled starting at 1
  triangles<pointT> T = readTrianglesFromFile<pointT>(triFile, 1);
  _seq<pointT> Pts    = readPointsFromFile<pointT>(rayFile);
  int n               = Pts.n / 2;
  ray<pointT>* rays   = newA(ray<pointT>, n);
  //  parallel_for (int i=0; i < n; i++) {
  parallel_doall(int, i, 0, n) {
    rays[i].o = Pts.A[2 * i];
    rays[i].d = Pts.A[2 * i + 1] - pointT(0, 0, 0);
  }
  parallel_doall_end timeRayCast(T, rays, n, rounds, oFile);
}
