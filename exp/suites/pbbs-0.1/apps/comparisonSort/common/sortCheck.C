// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2010 Guy Blelloch and the PBBS team
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
#include "quickSort.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;

struct strLess {
  bool operator() (char* s1c, char* s2c) {
    char* s1 = s1c, *s2 = s2c;
    while (*s1 && *s1==*s2) {s1++; s2++;};
    return (*s1 < *s2);
  }
};

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"<infile> <outfile>");
  pair<char*,char*> fnames = P.IOFileNames();
  seqData In = readSequenceFromFile(fnames.first);
  seqData Out = readSequenceFromFile(fnames.second);
  int n = In.n;
  elementType dt = In.dt;
  if (dt != Out.dt) {
    cout << "compSortCheck: types don't match" << endl;
    return(1);
  }
  if (n != Out.n) {
    cout << "compSortCheck: lengths dont' match" << endl;
    return(1);
  }

  if (dt == doubleT) {
    double* A = (double*) In.A;
    double* B = (double*) Out.A;
    compSort(A, n, less<double>());
    for(int i=0; i < n; i++) {
      if (A[i] != B[i]) {
	cout << "compSortCheck: check failed at i=" << i << endl;
	abort();
      }
    }
  } else if (dt == stringT) {
    char** A = (char**) In.A;
    char** B = (char**) Out.A;
    compSort(A, n, strLess());
    for(int i=0; i < n; i++) {
      if (strcmp(A[i],B[i])) {
	cout << "compSortCheck: check failed at i=" << i << endl;
	abort();
      }
    }
  } else {
    cout << "CompSortCheck: input files not of right type" << endl;
    return(1);
  }
}
