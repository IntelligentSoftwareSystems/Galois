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

// This code will generate (pseudo)-random points in the plummer distribution (3d)
//   or kuzmin distribution (2d)

#include <math.h>
#include "parallel.h"
#include "IO.h"
#include "parseCommandLine.h"
#include "geometry.h"
#include "geometryIO.h"
#include "geometryData.h"
#include "dataGen.h"
using namespace benchIO;
using namespace dataGen;
using namespace std;

 point2d randKuzmin(int i) {
   vect2d v = vect2d(randOnUnitSphere2d(i));
   int j = dataGen::hash<int>(i);
   double s = dataGen::hash<double>(j);
   double r = sqrt(1.0/((1.0-s)*(1.0-s))-1.0);
   return point2d(v*r);
 }

 point3d randPlummer(int i) {
   vect3d v = vect3d(randOnUnitSphere3d(i));
   int j = dataGen::hash<int>(i);
   double s = pow(dataGen::hash<double>(j),2.0/3.0);
   double r = sqrt(s/(1-s));
   return point3d(v*r);
 }

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"[-d {2,3}] n <outFile>");
  pair<int,char*> in = P.sizeAndFileName();
  int n = in.first;
  char* fname = in.second;
  int dims = P.getOptionIntValue("-d", 2);
  if (dims == 2) {
    point2d* Points = newA(point2d,n);
//    parallel_for (int i=0; i < n; i++) 
    parallel_doall(int, i, 0, n) {
      Points[i] = randKuzmin(i);
    } parallel_doall_end
    return writePointsToFile(Points,n,fname);
  } else if (dims == 3) {
    point3d* Points = newA(point3d,n);
//    parallel_for (int i=0; i < n; i++) 
    parallel_doall(int, i, 0, n) {
      Points[i] = randPlummer(i);
    } parallel_doall_end
    return writePointsToFile(Points,n,fname);
  } 
  P.badArgument();
  return 1;
}
