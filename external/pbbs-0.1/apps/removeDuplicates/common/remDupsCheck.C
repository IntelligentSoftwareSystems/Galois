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
#include "parallel.h"
#include "sequence.h"
#include "deterministicHash.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
#include "quickSort.h"
using namespace std;
using namespace benchIO;

bool strless(char* s1c, char* s2c) {
  char *s1 = s1c, *s2 = s2c;
  while (*s1 && *s1 == *s2) {
    s1++;
    s2++;
  };
  return (*s1 < *s2);
}

struct strLess {
  bool operator()(char* s1c, char* s2c) { return strless(s1c, s2c); }
};

struct strIntLess {
  bool operator()(stringIntPair* a, stringIntPair* b) {
    if (strless(a->first, b->first))
      return true;
    else if (strless(b->first, a->first))
      return false;
    else
      return (a->second < b->second);
  }
};

template <class T, class F>
void checkRemDups(T* A, int an, T* B, int bn, F less) {
  _seq<T> R = removeDuplicates(_seq<T>(A, an));
  compSort(R.A, R.n, less);

  if (R.n != bn) {
    cout << "removeDuplicatesCheck: check failed length test" << endl;
    abort();
  }
  compSort(B, bn, less);

  for (int i = 0; i < R.n; i++) {
    if (0) { // less(B[i],R.A[i]) || less(R.A[i],B[i])) {
      cout << "removeDuplicatesCheck: check failed equality test at i=" << i
           << endl;
      abort();
    }
  }
  R.del();
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "<inFile> <outFile>");
  pair<char*, char*> fnames = P.IOFileNames();
  seqData In                = readSequenceFromFile(fnames.first);
  seqData Out               = readSequenceFromFile(fnames.second);
  elementType dt            = In.dt;
  if (dt != Out.dt) {
    cout << "removeDuplicatesCheck: types don't match" << endl;
    return (1);
  }

  if (dt == intT) {
    // checkRemDups((int*) In.A, In.n, (int*) Out.A, Out.n, less<int>());
    In.del();
    Out.del();
  } else if (dt == stringT) {
    // checkRemDups((char**) In.A, In.n, (char**) Out.A, Out.n, strLess());
    In.del();
    Out.del();
  } else if (dt == stringIntPairT) {
    stringIntPair* AA = (stringIntPair*)In.A;
    stringIntPair** A = new stringIntPair*[In.n];
    //    parallel_for (int i=0; i < In.n; i++) A[i] = AA+i;
    parallel_doall(int, i, 0, In.n) { A[i] = AA + i; }
    parallel_doall_end stringIntPair* BB = (stringIntPair*)Out.A;
    stringIntPair** B                    = new stringIntPair*[In.n];
    //    parallel_for (int i=0; i < Out.n; i++) B[i] = BB+i;
    parallel_doall(int, i, 0, Out.n) { B[i] = BB + i; }
    parallel_doall_end checkRemDups(A, In.n, B, Out.n, strIntLess());
    In.del();
    Out.del();
    delete A;
  } else {
    cout << "removeDuplicatesCheck: input files not of right type" << endl;
    return (1);
  }
}
