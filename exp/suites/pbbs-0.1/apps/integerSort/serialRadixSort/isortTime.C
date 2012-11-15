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
#include "radixSort.h"
#include "gettime.h"
#include "parallel.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
#include "sequence.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
using namespace std;
using namespace benchIO;

typedef pair<uint,int> uintPair;

template <class T, class OT>
void timeIntegerSort(void* In, int n, int rounds, char* outFile) {
  T* A = (T*) In;
  T* B = new T[n];
  for (int j = 0; j < rounds; j++) {
//    parallel_for (int i=0; i < n; i++) B[i] = A[i];
    parallel_doall(int, i, 0, n) { B[i] = A[i]; } parallel_doall_end
    startTime();
    integerSort(B,n);
    nextTimeN();
  }
  cout << endl;
  if (outFile != NULL) 
    writeSequenceToFile((OT*) B, n, outFile);
  delete A;
  delete B;
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"[-o <outFile>] [-r <rounds>] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  int rounds = P.getOptionIntValue("-r",1);
  startTime();
  seqData D = readSequenceFromFile(iFile);
  elementType dt = D.dt;

  switch (dt) {
  case intT: 
    timeIntegerSort<uint,int>(D.A, D.n, rounds, oFile); break;
  case intPairT: 
    timeIntegerSort<uintPair,intPair>(D.A, D.n, rounds, oFile);  break;
  default:
    cout << "integer Sort: input file not of right type" << endl;
    return(1);
  }
}

