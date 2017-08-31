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
#include "nbody.h"
using namespace std;
using namespace benchIO;

// *************************************************************
//  TIMING
// *************************************************************

void timeNBody(point3d* pts, int n, int rounds, char* outFile) {
  particle** p = newA(particle*,n);
  particle* pp = newA(particle, n);
//  {parallel_for (int i=0; i < n; i++) 
  {
    parallel_doall(int, i, 0, n)  {
      p[i] = new (&pp[i]) particle(pts[i],1.0);
    } parallel_doall_end
  }
  for (int i=0; i < rounds; i++) {
    startTime();
    nbody(p, n);
    nextTimeN();
  }
  cout << endl;

  point3d* O = newA(point3d,n);
//  parallel_for(int i=0; i < n; i++) 
  parallel_doall(int, i, 0, n)  {
    O[i] = point3d(0.,0.,0.) + p[i]->force;
  } parallel_doall_end

  if (outFile != NULL) 
    writePointsToFile(O,n,outFile);

  free(O);
  free(p);
  free(pp);
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"[-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  int rounds = P.getOptionIntValue("-r",1);

  _seq<point3d> PIn = readPointsFromFile<point3d>(iFile);
  timeNBody(PIn.A, PIn.n, rounds, oFile);
}
