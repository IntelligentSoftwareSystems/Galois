#include <iostream>
#include <limits.h>
#include "graph.h"
using namespace std;

// Assumes root is negative
inline vindex find(vindex i, vindex* parent) {
  if ((parent[i]) < 0)
    return i;
  vindex j = parent[i];
  if (parent[j] < 0)
    return j;
  do
    j = parent[j];
  while (parent[j] >= 0);
  parent[i] = j;
  return j;
}

pair<int*, int> st(edgeArray EA) {
  edge* E         = EA.E;
  int m           = EA.nonZeros;
  int n           = EA.numRows;
  vindex* parents = newA(vindex, n);
  for (int i = 0; i < n; i++)
    parents[i] = -1;
  int* st   = newA(int, m);
  int nInSt = 0;
  for (int i = 0; i < m; i++) {
    vindex u = find(E[i].u, parents);
    vindex v = find(E[i].v, parents);
    if (u != v) {
      // union by rank -- join shallower
      // tree to deeper tree
      if (parents[v] < parents[u])
        swap(u, v);
      parents[u] += parents[v];
      parents[v]  = u;
      st[nInSt++] = i;
    }
  }
  free(parents);
  return pair<int*, int>(st, nInSt);
}
