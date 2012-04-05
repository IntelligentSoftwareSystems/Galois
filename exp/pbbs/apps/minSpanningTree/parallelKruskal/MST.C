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
#include <limits.h>
#include "sequence.h"
#include "graph.h"
#include "parallel.h"
#include "MST.h"
#include "gettime.h"
#include "speculative_for.h"
#include "unionFind.h"
using namespace std;

#if defined(CILK) || defined(CILKP)
#include "sampleSort.h"
#elif defined(OPENMP)
#include "stlParallelSort.h"
#else
#include "serialSort.h"
#endif

// **************************************************************
//    FIND OPERATION FOR UNION FIND
// **************************************************************

// Assumes root is negative
inline vindex find(vindex i, vindex* parent) {
  if ((parent[i]) < 0) return i;
  vindex j = parent[i];     
  if (parent[j] < 0) return j;
  do j = parent[j]; 
  while (parent[j] >= 0);
  vindex tmp;
  // shortcut all links on path
  while ((tmp = parent[i]) != j) { 
    parent[i] = j;
    i = tmp;
  }
  return j;
}

// **************************************************************
//    PARALLEL VERSION OF KRUSKAL'S ALGORITHM
// **************************************************************

struct indexedEdge {
  vindex u;
  vindex v;
  int id;
  indexedEdge(vindex _u, vindex _v, int _id) : u(_u), v(_v), id(_id) {}
};

struct UnionFindStep {
  int u;  int v;  
  indexedEdge *E;  reservation *R;  unionFind UF;  bool *inST;
  UnionFindStep(indexedEdge* _E, unionFind _UF, reservation* _R, bool* ist) 
    : E(_E), R(_R), UF(_UF), inST(ist) {}

  bool reserve(int i) {
    u = UF.find(E[i].u);
    v = UF.find(E[i].v);
    if (u != v) {
      R[v].reserve(i);
      R[u].reserve(i);
      return 1;
    } else return 0;
  }

  bool commit(int i) {
    if (R[v].check(i)) {
      R[u].checkReset(i); 
      UF.link(v, u); 
      inST[E[i].id] = 1;
      return 1;}
    else if (R[u].check(i)) {
      UF.link(u, v); 
      inST[E[i].id] = 1;
      return 1; }
    else return 0;
  }
};

template <class E, class F>
int almostKth(E* A, E* B, int k, int n, F f) {
  int ssize = min(1000,n);
  int stride = n/ssize;
  int km = (int) (k * ((double) ssize) / n);
  E T[ssize];
  for (int i = 0; i < ssize; i++) T[i] = A[i*stride];
  sort(T,T+ssize,f);
  E p = T[km];

  bool *flags = newA(bool,n);
//  {parallel_for (int i=0; i < n; i++) flags[i] = f(A[i],p);}
  {parallel_doall(int, i, 0, n) { flags[i] = f(A[i],p);} parallel_doall_end }
  int l = sequence::pack(A,B,flags,n);
//  {parallel_for (int i=0; i < n; i++) flags[i] = !flags[i];}
  {parallel_doall(int, i, 0, n) { flags[i] = !flags[i];} parallel_doall_end }
  sequence::pack(A,B+l,flags,n);
  free(flags);
  return l;
}

typedef std::pair<double,int> ei;

struct edgeLess {
  bool operator() (ei a, ei b) { 
    return (a.first == b.first) ? (a.second < b.second) 
      : (a.first < b.first);}};

pair<int*,int> mst(wghEdgeArray G) { 
  //startTime();
  wghEdge *E = G.E;
  ei* x = newA(ei,G.m);
//  parallel_for (int i=0; i < G.m; i++) 
  parallel_doall(int, i, 0, G.m) {
    x[i] = ei(E[i].weight,i);
  } parallel_doall_end
  //nextTime("copy with id");

  int l = min(4*G.n/3,G.m);
  ei* y = newA(ei,G.m);
  l = almostKth(x, y, l, G.m, edgeLess());
  //nextTime("kth smallest");

  compSort(y, l, edgeLess());
  //nextTime("first sort");

  unionFind UF(G.n);
  reservation *R = new reservation[G.n];
  //nextTime("initialize nodes");

  indexedEdge* z = newA(indexedEdge,G.m);
//  parallel_for (int i=0; i < l; i++) {
  parallel_doall(int, i, 0, l)  {
    int j = y[i].second;
    z[i] = indexedEdge(E[j].u,E[j].v,j);
  } parallel_doall_end
  //nextTime("copy to edges");

  bool *mstFlags = newA(bool, G.m);
//  parallel_for (int i=0; i < G.m; i++) mstFlags[i] = 0;
  parallel_doall(int, i, 0, G.m) { mstFlags[i] = 0; } parallel_doall_end
  UnionFindStep UFStep(z, UF, R,  mstFlags);
  speculative_for(UFStep, 0, l, 50);
  free(z);
  //nextTime("first union find loop");

  bool *flags = newA(bool,G.m-l);
//  parallel_for (int i = 0; i < G.m-l; i++) {
  parallel_doall(int, i, 0, G.m-l)  {
    int j = y[i+l].second;
    vindex u = UF.find(E[j].u);
    vindex v = UF.find(E[j].v);
    if (u != v) flags[i] = 1;
    else flags[i] = 0;
  } parallel_doall_end
  int k = sequence::pack(y+l, x, flags, G.m-l);
  free(flags);
  free(y);
  //nextTime("filter out self edges");

  compSort(x, k, edgeLess());
  //nextTime("second sort");

  z = newA(indexedEdge, k);
//  parallel_for (int i=0; i < k; i++) {
  parallel_doall(int, i, 0, k)  {
    int j = x[i].second;
    z[i] = indexedEdge(E[j].u,E[j].v,j);
  } parallel_doall_end
  free(x);
  //nextTime("copy to edges");

  UFStep = UnionFindStep(z, UF, R, mstFlags);
  speculative_for(UFStep, 0, k, 10);

  free(z); 
  //nextTime("second union find loop");

  int* mst = newA(int, G.m);
  int nInMst = sequence::packIndex(mst, mstFlags, G.m);
  free(mstFlags);
  //nextTime("pack results");

  //cout << "n=" << G.n << " m=" << G.m << " nInMst=" << nInMst << endl;
  UF.del(); delete R;
  return pair<int*,int>(mst, nInMst);
}
