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
#include "geometryIO.h"
#include "topology.h"
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;

bool checkDelaunay(tri *triangs, int n, int boundarySize);

#define MIN_ANGLE 30.0

bool skinnyTriangle(tri *t) {
  if (minAngleCheck(t->vtx[0]->pt, t->vtx[1]->pt, t->vtx[2]->pt, MIN_ANGLE))
    return 1;
  return 0;
}

double angle(tri *t) {
  return min(angle(t->vtx[0]->pt, t->vtx[1]->pt, t->vtx[2]->pt),
	     min(angle(t->vtx[1]->pt, t->vtx[0]->pt, t->vtx[2]->pt),
		 angle(t->vtx[2]->pt, t->vtx[0]->pt, t->vtx[1]->pt)));
}

bool check(triangles<point2d> Tri) {
  int m = Tri.numTriangles;
  vertex* V = NULL;
  tri* Triangs = NULL;
  topologyFromTriangles(Tri, &V, &Triangs);
  if (checkDelaunay(Triangs, m, 10)) return 1;
  int* bad = newA(int, m);
//  parallel_for (int i = 0; i < m; i++)
  parallel_doall(int, i, 0, m) {
    bad[i] = skinnyTriangle(&Triangs[i]);
  } parallel_doall_end
  int nbad = sequence::plusReduce(bad, m);
  if (nbad > 0) {
    cout << "Delaunay refine check: " << nbad << " skinny triangles" << endl;
    return 1;
  }
  return 0;
}

int parallel_main(int argc, char* argv[]) {
  commandLine P(argc,argv,
		"[-r <numtests>] <inFile> <outfile>");
  pair<char*,char*> fnames = P.IOFileNames();
  char* iFile = fnames.first;
  char* oFile = fnames.second;

  // number of random points to test
  int r = P.getOptionIntValue("-r",10);

  triangles<point2d> T = readTrianglesFromFile<point2d>(oFile,0);
  return check(T);
}
