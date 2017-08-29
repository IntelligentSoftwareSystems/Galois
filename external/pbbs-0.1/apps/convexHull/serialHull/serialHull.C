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
#include "utils.h"
#include "geometry.h"
#include "hull.h"
using namespace std;

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

_seq<int> hull(point2d* P, int n) {
  int* I = newA(int, n);
  for (int i=0; i < n; i++) I[i] = i;

  int l = 0;
  int r = 0;
  for (int i=1; i < n; i++) {
    if (P[i].x > P[r].x) r = i;
    if (P[i].x < P[l].x || ((P[i].x == P[l].x) && P[i].y < P[l].y)) 
      l = i;
  }

  pair<int,int> nn = split(I, n, aboveLine(P, l, r), aboveLine(P, r, l));
  int n1 = nn.first;
  int n2 = nn.second;

  int m1 = serialQuickHull(I, P, n1, l, r);
  int m2 = serialQuickHull(I+n-n2, P, n2, r, l);

  for (int i=m1; i > 0; i--) I[i] = I[i-1];
  for (int i=0; i < m2; i++) I[i+m1+2] = I[i+n-n2];
  I[0] = l;
  I[m1+1] = r;
  return _seq<int>(I,m1+2+m2);
}

