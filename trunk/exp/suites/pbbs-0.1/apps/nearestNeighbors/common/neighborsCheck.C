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
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;

template <class pointT>
int checkNeighbors(_seq<int> neighbors, pointT* P, int n, int k, int r) {
  if (neighbors.n != k * n) {
    cout << "error in neighborsCheck: wrong length, n = " << n 
	 << " k = " << k << " neighbors = " << neighbors.n << endl;
    return 1;
  }

  for (int j = 0; j < r; j++) {
    int jj = utils::hash(j) % n;

    double* distances = newA(double,n);
//    parallel_for (int i=0; i < n; i++) {
    parallel_doall(int, i, 0, n)  {
      if (i == jj) distances[i] = DBL_MAX;
      else distances[i] = (P[jj] - P[i]).Length();
    } parallel_doall_end
    double minD = sequence::reduce<double>(distances, n, utils::minF<double>());

    double d = (P[jj] - P[(neighbors.A)[k*jj]]).Length();

    double errorTolerance = 1e-6;
    if ((d - minD) / (d + minD)  > errorTolerance) {
      cout << "error in neighborsCheck: for point " << jj 
	   << " min distance reported is: " << d 
	   << " actual is: " << minD << endl;
      return 1;
    }
  }
  return 0;
}


int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,
		"[-k {1,...,10}] [-d {2,3}] [-r <numtests>] <inFile> <outfile>");
  pair<char*,char*> fnames = P.IOFileNames();
  char* iFile = fnames.first;
  char* oFile = fnames.second;

  // number of random points to test
  int r = P.getOptionIntValue("-r",10);

  int k = P.getOptionIntValue("-k",1);
  int d = P.getOptionIntValue("-d",2);
  if (k > 10 || k < 1) P.badArgument();
  if (d < 2 || d > 3) P.badArgument();

  _seq<int> neighbors = readIntArrayFromFile(oFile);

  if (d == 2) {
    _seq<point2d> PIn = readPointsFromFile<point2d>(iFile);
    int n = PIn.n;
    point2d* P = PIn.A;
    return checkNeighbors(neighbors, PIn.A, PIn.n, k, r);
  } else if (d == 3) {
    _seq<point3d> PIn = readPointsFromFile<point3d>(iFile);
    int n = PIn.n;
    point3d* P = PIn.A;
    return checkNeighbors(neighbors, PIn.A, PIn.n, k, r);
  } else return 1;
}
