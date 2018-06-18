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
#include "IO.h"
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;

typedef unsigned char uchar;

bool strLessBounded(uchar* s1c, uchar* s2c, int n) {
  uchar *s1 = s1c, *s2 = s2c;
  while (*s1 && *s1 == *s2) {
    if (n-- < 0)
      return 1;
    s1++;
    s2++;
  };
  return (*s1 < *s2);
}

bool isPermutation(int* SA, int n) {
  int* seen = newA(int, n);
  //  parallel_for (int i = 0;  i < n;  i++) seen[i] = 0;
  parallel_doall(int, i, 0, n) { seen[i] = 0; }
  parallel_doall_end
  //  parallel_for (int i = 0;  i < n;  i++) seen[SA[i]] = 1;
  parallel_doall(int, i, 0, n) {
    seen[SA[i]] = 1;
  }
  parallel_doall_end int nseen = sequence::plusReduce(seen, n);
  return (nseen == n);
}

bool isSorted(int* SA, uchar* s, int n) {
  int checkLen = 100;
  int error    = n;
  //  parallel_for (int i = 0;  i < n-1;  i++)
  parallel_doall(int, i, 0, n - 1) {
    if (!strLessBounded(s + SA[i], s + SA[i + 1], checkLen)) {
      // cout.write((char*) s+SA[i],checkLen); cout << endl;
      // cout.write((char*) s+SA[i+1],min(checkLen,n-SA[i+1]));cout << endl;
      utils::writeMin(&error, i);
    }
  }
  parallel_doall_end if (error != n) {
    cout << "Suffix Array Check: not sorted at i = " << error + 1 << endl;
    return 0;
  }
  return 0;
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "<infile> <outfile>");
  pair<char*, char*> fnames = P.IOFileNames();
  _seq<char> In             = readStringFromFile(fnames.first);
  uchar* s                  = (uchar*)In.A;
  _seq<int> Out             = readIntArrayFromFile(fnames.second);
  if (In.n != Out.n) {
    cout << "Suffix Array Check: lengths don't match" << endl;
    return 1;
  }
  if (!isPermutation(Out.A, Out.n)) {
    cout << "Suffix Array Check: array is not a permutation" << endl;
    return 1;
  }
  return !isSorted(Out.A, s, Out.n);
}
