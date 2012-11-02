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
#include "blockRadixSort.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;

typedef pair<uint,int> uintPair;

template <class T>
void checkIntegerSort(void* In, void* Out, int n) {
  T* A = (T*) In;
  T* B = (T*) Out;
  integerSort(A, n);
  for(int i=0; i < n; i++) {
    if (A[i] != B[i]) {
      cout << "integerSortCheck: check failed at i=" << i << endl;
      abort();
    }
  }
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"<inFile> <outFile>");
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

  switch (dt) {
  case intT: checkIntegerSort<uint>(In.A, Out.A, n); break;
  case intPairT: checkIntegerSort<uintPair>(In.A, Out.A, n); break;
  default:
    cout << "integerSortCheck: input files not of right type" << endl;
    return(1);
  }
}
