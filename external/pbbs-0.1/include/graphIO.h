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

#ifndef _BENCH_GRAPH_IO
#define _BENCH_GRAPH_IO

#include "parallel.h"
#include "IO.h"

using namespace benchIO;

int xToStringLen(edge a) { return xToStringLen(a.u) + xToStringLen(a.v) + 1; }

void xToString(char* s, edge a) {
  int l = xToStringLen(a.u);
  xToString(s, a.u);
  s[l] = ' ';
  xToString(s + l + 1, a.v);
}

int xToStringLen(wghEdge a) {
  return xToStringLen(a.u) + xToStringLen(a.v) + xToStringLen(a.weight) + 2;
}

void xToString(char* s, wghEdge a) {
  int lu = xToStringLen(a.u);
  int lv = xToStringLen(a.v);
  xToString(s, a.u);
  s[lu] = ' ';
  xToString(s + lu + 1, a.v);
  s[lu + lv + 1] = ' ';
  xToString(s + lu + lv + 2, a.weight);
}

namespace benchIO {
using namespace std;

string AdjGraphHeader     = "AdjacencyGraph";
string EdgeArrayHeader    = "EdgeArray";
string WghEdgeArrayHeader = "WeightedEdgeArray";

int writeGraphToFile(graph G, char* fname) {
  int m        = G.m;
  int n        = G.n;
  int totalLen = 2 + n + m;
  int* Out     = newA(int, totalLen);
  Out[0]       = n;
  Out[1]       = m;
  //    parallel_for (int i=0; i < n; i++) {
  parallel_doall(int, i, 0, n) { Out[i + 2] = G.V[i].degree; }
  parallel_doall_end int total =
      sequence::scan(Out + 2, Out + 2, n, utils::addF<int>(), 0);
  for (int i = 0; i < n; i++) {
    int* O   = Out + (2 + n + Out[i + 2]);
    vertex v = G.V[i];
    for (int j = 0; j < v.degree; j++)
      O[j] = v.Neighbors[j];
  }
  int r = writeArrayToFile(AdjGraphHeader, Out, totalLen, fname);
  free(Out);
  return r;
}

int writeEdgeArrayToFile(edgeArray EA, char* fname) {
  int m = EA.nonZeros;
  int r = writeArrayToFile(EdgeArrayHeader, EA.E, m, fname);
  return r;
}

int writeWghEdgeArrayToFile(wghEdgeArray EA, char* fname) {
  int m = EA.m;
  int r = writeArrayToFile(WghEdgeArrayHeader, EA.E, m, fname);
  return r;
}

edgeArray readEdgeArrayFromFile(char* fname) {
  _seq<char> S = readStringFromFile(fname);
  words W      = stringToWords(S.A, S.n);
  if (W.Strings[0] != EdgeArrayHeader) {
    cout << "Bad input file" << endl;
    abort();
  }
  long n  = (W.m - 1) / 2;
  edge* E = newA(edge, n);
  //    {parallel_for(long i=0; i < n; i++)
  {
    parallel_doall(long, i, 0, n) {
      E[i] = edge(atoi(W.Strings[2 * i + 1]), atoi(W.Strings[2 * i + 2]));
    }
    parallel_doall_end
  }

  int maxR = 0;
  int maxC = 0;
  for (int i = 0; i < n; i++) {
    maxR = max(maxR, E[i].u);
    maxC = max(maxC, E[i].v);
  }
  return edgeArray(E, maxR + 1, maxC + 1, n);
}

wghEdgeArray readWghEdgeArrayFromFile(char* fname) {
  _seq<char> S = readStringFromFile(fname);
  words W      = stringToWords(S.A, S.n);
  if (W.Strings[0] != WghEdgeArrayHeader) {
    cout << "Bad input file" << endl;
    abort();
  }
  long n     = (W.m - 1) / 3;
  wghEdge* E = newA(wghEdge, n);
  //    {parallel_for(long i=0; i < n; i++)
  {
    parallel_doall(long, i, 0, n) {
      E[i] = wghEdge(atoi(W.Strings[3 * i + 1]), atoi(W.Strings[3 * i + 2]),
                     atof(W.Strings[3 * i + 3]));
    }
    parallel_doall_end
  }
  int maxR = 0;
  int maxC = 0;
  for (int i = 0; i < n; i++) {
    maxR = max(maxR, E[i].u);
    maxC = max(maxC, E[i].v);
  }
  return wghEdgeArray(E, max(maxR, maxC) + 1, n);
}

graph readGraphFromFile(char* fname) {
  _seq<char> S = readStringFromFile(fname);
  words W      = stringToWords(S.A, S.n);
  if (W.Strings[0] != AdjGraphHeader) {
    cout << "Bad input file" << endl;
    abort();
  }
  int len = W.m - 1;
  int* In = newA(int, len);
  //    {parallel_for(long i=0; i < len; i++) In[i] = atoi(W.Strings[i + 1]);}
  {
    parallel_doall(long, i, 0, len) { In[i] = atoi(W.Strings[i + 1]); }
    parallel_doall_end
  }
  int n = In[0];
  int m = In[1];
  if (len != n + m + 2) {
    cout << "Bad input file" << endl;
    abort();
  }
  vertex* v    = newA(vertex, n);
  int* offsets = In + 2;
  int* edges   = In + 2 + n;
  //    parallel_for (int i=0; i < n; i++) {
  parallel_doall(int, i, 0, n) {
    int o          = offsets[i];
    int l          = ((i == n - 1) ? m : offsets[i + 1]) - offsets[i];
    v[i].degree    = l;
    v[i].Neighbors = edges + o;
  }
  parallel_doall_end return graph(v, n, m, In);
}

}; // namespace benchIO

#endif // _BENCH_GRAPH_IO
