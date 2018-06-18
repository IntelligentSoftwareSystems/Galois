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
#include <cstring>
#include "parallel.h"
#include "geometry.h"
#include "geometryIO.h"
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;

struct getX {
  point2d* P;
  getX(point2d* _P) : P(_P) {}
  double operator()(int i) { return P[i].x; }
};

struct lessX {
  bool operator()(point2d a, point2d b) {
    return (a.x < b.x) ? 1 : (a.x > b.x) ? 0 : (a.y < b.y);
  }
};

bool eq(point2d a, point2d b) { return (a.x == b.x) && (a.y == b.y); }

bool checkHull(_seq<point2d> PIn, _seq<int> I) {
  point2d* P  = PIn.A;
  int n       = PIn.n;
  int nOut    = I.n;
  point2d* PO = newA(point2d, nOut);
  for (int i = 0; i < nOut; i++)
    PO[i] = P[I.A[i]];
  int idx = sequence::maxIndex<double>(0, nOut, greater<double>(), getX(PO));
  sort(P, P + n, lessX());
  if (!eq(P[0], PO[0])) {
    cout << "checkHull: bad leftmost point" << endl;
    P[0].print();
    PO[0].print();
    cout << endl;
    return 1;
  }
  if (!eq(P[n - 1], PO[idx])) {
    cout << "checkHull: bad rightmost point" << endl;
    return 1;
  }
  int k = 1;
  for (int i = 0; i < idx; i++) {
    if (i > 0 && counterClockwise(PO[i - 1], PO[i], P[i + 1])) {
      cout << "checkHull: not convex" << endl;
      return 1;
    }
    if (PO[i].x > PO[i + 1].x) {
      cout << "checkHull: not sorted by x" << endl;
      return 1;
    }
    while (!eq(P[k], PO[i + 1]) && k < n)
      if (counterClockwise(PO[i], PO[i + 1], P[k++])) {
        cout << "checkHull: above hull" << endl;
        return 1;
      }
    if (k == n) {
      cout << "checkHull: unexpected points in hull" << endl;
      return 1;
    }
    k++;
  }
  free(PO);
  return 0;
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "<inFile> <outfile>");
  pair<char*, char*> fnames = P.IOFileNames();
  char* iFile               = fnames.first;
  char* oFile               = fnames.second;

  _seq<point2d> PIn = readPointsFromFile<point2d>(iFile);
  _seq<int> I       = readIntArrayFromFile(oFile);
  return checkHull(PIn, I);
}
