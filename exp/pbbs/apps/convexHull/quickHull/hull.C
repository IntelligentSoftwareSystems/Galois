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

#include <algorithm>
#include "parallel.h"
#include "geometry.h"
#include "sequence.h"
using namespace std;
using namespace sequence;

template <class ET, class F>
pair<int,int> split(ET* A, int n, F lf, F rf) {
  int ll = 0, lm = 0;
  int rm = n-1, rr = n-1;
  while (1) {
    while ((lm <= rm) && !(rf(A[lm]) > 0)) {
      if (lf(A[lm]) > 0) A[ll++] = A[lm];
      lm++;
    }
    while ((rm >= lm) && !(lf(A[rm]) > 0)) {
      if (rf(A[rm]) > 0) A[rr--] = A[rm];
      rm--;
    }
    if (lm >= rm) break; 
    ET tmp = A[lm++];
    A[ll++] = A[rm--];
    A[rr--] = tmp;
  }
  int n1 = ll;
  int n2 = n-rr-1;
  return pair<int,int>(n1,n2);
}

struct aboveLine {
  int l, r;
  point2d* P;
  aboveLine(point2d* _P, int _l, int _r) : P(_P), l(_l), r(_r) {}
  bool operator() (int i) {return triArea(P[l], P[r], P[i]) > 0.0;}
};

int serialQuickHull(int* I, point2d* P, int n, int l, int r) {
  if (n < 2) return n;
  int maxP = I[0];
  double maxArea = triArea(P[l],P[r],P[maxP]);
  for (int i=1; i < n; i++) {
    int j = I[i];
    double a = triArea(P[l],P[r],P[j]);
    if (a > maxArea) {
      maxArea = a;
      maxP = j;
    }
  }

  pair<int,int> nn = split(I, n, aboveLine(P,l,maxP), aboveLine(P,maxP,r));
  int n1 = nn.first;
  int n2 = nn.second;

  int m1, m2;
  m1 = serialQuickHull(I,      P, n1, l,   maxP);
  m2 = serialQuickHull(I+n-n2, P, n2, maxP,r);
  for (int i=0; i < m2; i++) I[i+m1+1] = I[i+n-n2];
  I[m1] = maxP;
  return m1+1+m2;
}

struct triangArea {
  int l, r;
  point2d* P;
  int* I;
  triangArea(int* _I, point2d* _P, int _l, int _r) : I(_I), P(_P), l(_l), r(_r) {}
  double operator() (int i) {return triArea(P[l], P[r], P[I[i]]);}
};

int quickHull(int* I, int* Itmp, point2d* P, int n, int l, int r, int depth) {
  if (n < 2) // || depth == 0) 
    return serialQuickHull(I, P, n, l, r);
  else {
    
    int idx = maxIndex<double>(0,n,greater<double>(),triangArea(I,P,l,r));
    int maxP = I[idx];

    int n1 = filter(I, Itmp,    n, aboveLine(P, l, maxP));
    int n2 = filter(I, Itmp+n1, n, aboveLine(P, maxP, r));

    int m1, m2;
    m1 = cilk_spawn quickHull(Itmp, I ,P, n1, l, maxP, depth-1);
    m2 = quickHull(Itmp+n1, I+n1, P, n2, maxP, r, depth-1);
    cilk_sync;

//    parallel_for (int i=0; i < m1; i++) I[i] = Itmp[i];
    parallel_doall(int, i, 0, m1) { I[i] = Itmp[i]; } parallel_doall_end
    I[m1] = maxP;
//    parallel_for (int i=0; i < m2; i++) I[i+m1+1] = Itmp[i+n1];
    parallel_doall(int, i, 0, m2) { I[i+m1+1] = Itmp[i+n1]; } parallel_doall_end
    return m1+1+m2;
  }
}

struct makePair {
  pair<int,int> operator () (int i) { return pair<int,int>(i,i);}
};

struct minMaxIndex {
  point2d* P;
  minMaxIndex (point2d* _P) : P(_P) {}
  pair<int,int> operator () (pair<int,int> l, pair<int,int> r) {
    int minIndex = 
      (P[l.first].x < P[r.first].x) ? l.first :
      (P[l.first].x > P[r.first].x) ? r.first :
      (P[l.first].y < P[r.first].y) ? l.first : r.first;
    int maxIndex = (P[l.second].x > P[r.second].x) ? l.second : r.second;
    return pair<int,int>(minIndex, maxIndex);
  }
};
    
_seq<int> hull(point2d* P, int n) {
  pair<int,int> minMax = reduce<pair<int,int> >(0,n,minMaxIndex(P), makePair());
  int l = minMax.first;
  int r = minMax.second;
  bool* fTop = newA(bool,n);
  bool* fBot = newA(bool,n);
  int* I = newA(int, n);
  int* Itmp = newA(int, n);
//  parallel_for(int i=0; i < n; i++) {
  parallel_doall(int, i, 0, n)  {
    Itmp[i] = i;
    double a = triArea(P[l],P[r],P[i]);
    fTop[i] = a > 0;
    fBot[i] = a < 0;
  } parallel_doall_end

  int n1 = pack(Itmp, I, fTop, n);
  int n2 = pack(Itmp, I+n1, fBot, n);
  free(fTop); free(fBot);

  int m1; int m2;
  m1 = cilk_spawn quickHull(I, Itmp, P, n1, l, r, 5);
  m2 = quickHull(I+n1, Itmp+n1, P, n2, r, l, 5);
  cilk_sync;

//  parallel_for (int i=0; i < m1; i++) Itmp[i+1] = I[i];
  parallel_doall(int, i, 0, m1) { Itmp[i+1] = I[i]; } parallel_doall_end
//  parallel_for (int i=0; i < m2; i++) Itmp[i+m1+2] = I[i+n1];
  parallel_doall(int, i, 0, m2) { Itmp[i+m1+2] = I[i+n1]; } parallel_doall_end
  free(I);
  Itmp[0] = l;
  Itmp[m1+1] = r;
  return _seq<int>(Itmp, m1+2+m2);
}

