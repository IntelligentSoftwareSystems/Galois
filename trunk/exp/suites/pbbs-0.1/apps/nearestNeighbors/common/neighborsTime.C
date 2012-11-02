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
using namespace std;
using namespace benchIO;

// *************************************************************
//  SOME DEFINITIONS
// *************************************************************

#define K 10

template <class PT, int KK>
struct vertex {
  typedef PT pointT;
  int identifier;
  pointT pt;         // the point itself
  vertex* ngh[KK];    // the list of neighbors
  vertex(pointT p, int id) : pt(p), identifier(id) {}

};

// *************************************************************
//  TIMING
// *************************************************************

template <int maxK, class point>
void timeNeighbors(point* pts, int n, int k, int rounds, char* outFile) {
  typedef vertex<point,maxK> vertex;
  int dimensions = pts[0].dimension();
  vertex** v = newA(vertex*,n);
  vertex* vv = newA(vertex, n);
//  {parallel_for (int i=0; i < n; i++) 
  {
    parallel_doall(int, i, 0, n) {
      v[i] = new (&vv[i]) vertex(pts[i],i);
    } parallel_doall_end
  }
  for (int i=0; i < rounds; i++) {
    startTime();
    ANN<maxK>(v, n, k);
    nextTimeN();
  }
  cout << endl;

  int m = n * k;
  int* Pout = newA(int, m);
//  {parallel_for (int i=0; i < n; i++) {
  {
    parallel_doall(int, i, 0, n)  {
      for (int j=0; j < k; j++)
        Pout[maxK*i + j] = (v[i]->ngh[j])->identifier;
    } parallel_doall_end
  }
  if (outFile != NULL) writeIntArrayToFile(Pout, m, outFile);

  free(Pout);
  free(v);
  free(vv);
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"[-k {1,...,10}] [-d {2,3}] [-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  int rounds = P.getOptionIntValue("-r",1);
  int k = P.getOptionIntValue("-k",1);
  int d = P.getOptionIntValue("-d",2);
  if (k > 10 || k < 1) P.badArgument();
  if (d < 2 || d > 3) P.badArgument();

  if (d == 2) {
    _seq<point2d> PIn = readPointsFromFile<point2d>(iFile);
    if (k == 1) timeNeighbors<1>(PIn.A, PIn.n, 1, rounds, oFile);
    else timeNeighbors<10>(PIn.A, PIn.n, k, rounds, oFile);
  }

  if (d == 3) {
    _seq<point3d> PIn = readPointsFromFile<point3d>(iFile);
    if (k == 1) timeNeighbors<1>(PIn.A, PIn.n, 1, rounds, oFile);
    else timeNeighbors<10>(PIn.A, PIn.n, k, rounds, oFile);
  }

}
