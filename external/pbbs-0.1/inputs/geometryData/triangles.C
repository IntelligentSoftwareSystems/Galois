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

#include <math.h>
#include "parallel.h"
#include "IO.h"
#include "geometry.h"
#include "geometryIO.h"
#include "geometryData.h"
#include "dataGen.h"
#include "parseCommandLine.h"
using namespace benchIO;
using namespace dataGen;
using namespace std;

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "n <outFile>");
  pair<int, char*> in = P.sizeAndFileName();
  bool onSphere       = P.getOption("-S");

  int n               = in.first;
  char* fname         = in.second;
  point3d* Points     = newA(point3d, n * 3);
  triangle* Triangles = newA(triangle, n);
  double d            = 1.0 / sqrt((double)n);
  //  parallel_for (int i=0; i < n; i++) {
  parallel_doall(int, i, 0, n) {
    if (onSphere)
      Points[3 * i] = randOnUnitSphere3d(i);
    else
      Points[3 * i] = rand3d(i);
    Points[3 * i + 1] = Points[3 * i] + vect3d(d, d, 0);
    Points[3 * i + 2] = Points[3 * i] + vect3d(d, 0, d);
    Triangles[i].C[0] = 3 * i;
    Triangles[i].C[1] = 3 * i + 1;
    Triangles[i].C[2] = 3 * i + 2;
  }
  parallel_doall_end return writeTrianglesToFile(
      triangles<point3d>(3 * n, n, Points, Triangles), fname);
}
