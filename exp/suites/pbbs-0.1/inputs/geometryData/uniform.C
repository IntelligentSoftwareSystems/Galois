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

// This code will generate uniformly (pseudo)-random points
//   By default they will be in the unit cube [-1..1] in each dimension
//   The -s argument will place them in a unit sphere centered at 0 with
//      unit radius
//   The -S argument will place them on the surface of the unit sphere
//   Only one of -s or -S should be used

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
  commandLine P(argc,argv,"[-s] [-S] [-d {2,3}] n <outFile>");
  pair<int,char*> in = P.sizeAndFileName();
  int n = in.first;
  char* fname = in.second;
  int dims = P.getOptionIntValue("-d", 2);
  bool inSphere = P.getOption("-s");
  bool onSphere = P.getOption("-S");
  if (dims == 2) {
    point2d* Points = newA(point2d,n);
//    parallel_for (int i=0; i < n; i++) 
    parallel_doall(int, i, 0, n) {
      if (inSphere) Points[i] = randInUnitSphere2d(i);
      else if (onSphere) Points[i] = randOnUnitSphere2d(i);
      else Points[i] = rand2d(i);
    } parallel_doall_end
    return writePointsToFile(Points,n,fname);
  } else if (dims == 3) {
    point3d* Points = newA(point3d,n);
//    parallel_for (int i=0; i < n; i++) 
    parallel_doall(int, i, 0, n) {
      if (inSphere) Points[i] = randInUnitSphere3d(i);
      else if (onSphere) Points[i] = randOnUnitSphere3d(i);
      else Points[i] = rand3d(i);
    } parallel_doall_end
    return writePointsToFile(Points,n,fname);
  } 
  P.badArgument();
  return 1;
}
