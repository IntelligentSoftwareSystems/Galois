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

struct minpt {point3d operator() (point3d a, point3d b) {return a.minCoords(b);}};
struct maxpt {point3d operator() (point3d a, point3d b) {return a.maxCoords(b);}};

// p0 is lower corner
// p1 is upper corner
ray<point3d>* generateRays(point3d p0, point3d p1, int n) {
  vect3d d = p1 - p0;
  ray<point3d>* rays = newA(ray<point3d>,n);
  // parallel for seems to break
  for (int i=0; i < n; i++) {
    point3d pl = point3d(p0.x + d.x * dataGen::hash<double>(4*i+0), 
			 p0.y + d.y * dataGen::hash<double>(4*i+1), 
			 p0.z);
    point3d pr = point3d(p0.x + d.x * dataGen::hash<double>(4*i+2), 
			 p0.y + d.y * dataGen::hash<double>(4*i+3), 
			 p1.z);
    rays[i] = ray<point3d>(pl, pr-pl);
  }
  return rays;
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"[-n <num>] <triangleInFile> <rayOutFile>");
  pair<char*,char*> fnames = P.IOFileNames();
  char* ifile = fnames.first;
  char* ofile = fnames.second;

  triangles<point3d> T = readTrianglesFromFile<point3d>(ifile,1);
  int n = P.getOptionIntValue("-n", T.numTriangles);

  point3d minPt = sequence::reduce(T.P, T.numPoints, minpt());
  point3d maxPt = sequence::reduce(T.P, T.numPoints, maxpt());

  // generate as many rays as triangles
  ray<point3d>* rays = generateRays(minPt, maxPt, T.numTriangles);
  point3d* pts = newA(point3d, 2*n);
//  parallel_for(int i=0; i < n; i++) {
  parallel_doall(int, i, 0, n)  {
    pts[2*i] = rays[i].o;
    pts[2*i+1] = point3d(0,0,0) + rays[i].d;
  } parallel_doall_end
  return writePointsToFile(pts, 2*n, ofile);
}
