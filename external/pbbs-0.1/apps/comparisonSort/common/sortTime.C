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
#include "gettime.h"
#include "utils.h"
#include "randPerm.h"
#include "parallel.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;

struct strCmp {
  bool operator()(char* s1c, char* s2c) {
    char *s1 = s1c, *s2 = s2c;
    while (*s1 && *s1 == *s2) {
      s1++;
      s2++;
    };
    return (*s1 < *s2);
  }
};

template <class T, class CMP>
void timeSort(T* A, int n, CMP f, int rounds, bool permute, char* outFile) {
  if (permute)
    randPerm(A, n);
  T* B = new T[n];
  //  parallel_for (int i=0; i < n; i++) B[i] = A[i];
  parallel_doall(int, i, 0, n) { B[i] = A[i]; }
  parallel_doall_end compSort(B, n, f); // run one sort to "warm things up"
  for (int i = 0; i < rounds; i++) {
    //    parallel_for (int i=0; i < n; i++) B[i] = A[i];
    parallel_doall(int, i, 0, n) { B[i] = A[i]; }
    parallel_doall_end startTime();
    compSort(B, n, f);
    nextTimeN();
  }
  cout << endl;
  if (outFile != NULL)
    writeSequenceToFile(B, n, outFile);
  delete B;
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "[-p] [-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile  = P.getArgument(0);
  char* oFile  = P.getOptionValue("-o");
  int rounds   = P.getOptionIntValue("-r", 1);
  bool permute = P.getOption("-p");
  seqData D    = readSequenceFromFile(iFile);
  int dt       = D.dt;

  switch (dt) {
  case doubleT:
    timeSort((double*)D.A, D.n, less<double>(), rounds, permute, oFile);
    break;
  case stringT:
    timeSort((char**)D.A, D.n, strCmp(), rounds, permute, oFile);
    break;
  default:
    cout << "comparisonSort: input file not of right type" << endl;
    return (1);
  }
}
