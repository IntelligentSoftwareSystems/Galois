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

#include "parallel.h"
#include "util.h"

struct unionFind {
  int* parents;

  // initialize with all roots marked with -1
  unionFind(int n) {
    parents = newA(int, n);
    //    parallel_for(int i=0; i < n; i++) parents[i] = -1;
    parallel_doall(int, i, 0, n) { parents[i] = -1; }
    parallel_doall_end
  }

  void del() { free(parents); }

  vindex find(vindex i) {
    if (parents[i] < 0)
      return i;
    vindex j = parents[i];
    if (parents[j] < 0)
      return j;
    do
      j = parents[j];
    while (parents[j] >= 0);
    vindex tmp;
    while ((tmp = parents[i]) != j) {
      parents[i] = j;
      i          = tmp;
    }
    return j;
  }

  void link(vindex u, vindex v) { parents[u] = v; }
};
