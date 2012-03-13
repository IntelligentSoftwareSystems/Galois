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
#include "float.h"
#include <algorithm>
#include <cstring>
#include "parallel.h"
#include "geometry.h"
#include "sequence.h"
#include "topology.h"
#include "geometryIO.h"
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;


bool checkDelaunay(tri *triangs, int n, int boundarySize);

bool check(triangles<point2d> Tri, _seq<point2d> P) {
  int m = Tri.numTriangles;
  for (int i=0; i < P.n; i++)
    if (P.A[i].x != Tri.P[i].x || P.A[i].y != Tri.P[i].y) {
      cout << "checkDelaunay: prefix of points don't match input" << endl;
      return 0;
    }
  vertex* V = NULL;
  tri* Triangs = NULL;
  topologyFromTriangles(Tri, &V, &Triangs);
  return checkDelaunay(Triangs, m, 10);
}
    

int parallel_main(int argc, char* argv[]) {
  commandLine P(argc,argv,
		"[-r <numtests>] <inFile> <outfile>");
  pair<char*,char*> fnames = P.IOFileNames();
  char* iFile = fnames.first;
  char* oFile = fnames.second;

  // number of random points to test
  int r = P.getOptionIntValue("-r",10);

  _seq<point2d> PIn = readPointsFromFile<point2d>(iFile);
  triangles<point2d> T = readTrianglesFromFile<point2d>(oFile,0);
  check(T, PIn);

  return 0;
}
