#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include "parallel.h"
#include "graphUtils.h"
#include "sequence.h"
#include "blockRadixSort.h"
#include "deterministicHash.h"

using namespace std;

wghEdgeArray addRandWeights(edgeArray G) {
  int m      = G.nonZeros;
  int n      = G.numRows;
  wghEdge* E = newA(wghEdge, m);
  for (int i = 0; i < m; i++) {
    E[i].u      = G.E[i].u;
    E[i].v      = G.E[i].v;
    E[i].weight = utils::hashInt(i);
  }
  return wghEdgeArray(E, n, m);
}

edgeArray edgesFromSparse(sparseRowMajorD M) {
  edge* E = newA(edge, M.nonZeros);
  int k   = 0;
  for (int i = 0; i < M.numRows; i++) {
    for (int j = M.Starts[i]; j < M.Starts[i + 1]; j++) {
      if (M.Values[j] != 0.0) {
        E[k].u = i;
        E[k].v = M.ColIds[j];
        k++;
      }
    }
  }
  int nonZeros = k;
  return edgeArray(E, M.numRows, M.numCols, nonZeros);
}

int cmpInt(int v, int b) { return (v > b) ? 1 : ((v == b) ? 0 : -1); }

struct hashEdge {
  typedef edge* eType;
  typedef edge* kType;
  eType empty() { return NULL; }
  kType getKey(eType v) { return v; }
  unsigned int hash(kType e) {
    return utils::hashInt(e->u) + utils::hashInt(100 * e->v);
  }
  int cmp(kType a, kType b) {
    int c = cmpInt(a->u, b->u);
    return (c == 0) ? cmpInt(a->v, b->v) : c;
  }
  bool replaceQ(eType v, eType b) { return 0; }
};

_seq<edge*> removeDuplicates(_seq<edge*> S) {
  return removeDuplicates(S, hashEdge());
}

edgeArray remDuplicates(edgeArray A) {
  int m     = A.nonZeros;
  edge** EP = newA(edge*, m);
  //  parallel_for (int i=0;i < m; i++) EP[i] = A.E+i;
  parallel_doall(int, i, 0, m) { EP[i] = A.E + i; }
  parallel_doall_end _seq<edge*> F = removeDuplicates(_seq<edge*>(EP, m));
  free(EP);
  int l   = F.n;
  edge* E = newA(edge, m);
  //  parallel_for (int i=0; i < l; i++) E[i] = *F.A[i];
  parallel_doall(int, i, 0, l) { E[i] = *F.A[i]; }
  parallel_doall_end F.del();
  return edgeArray(E, A.numRows, A.numCols, l);
}

struct nEQF {
  bool operator()(edge e) { return (e.u != e.v); }
};

edgeArray makeSymmetric(edgeArray A) {
  int m   = A.nonZeros;
  edge* E = A.E;
  edge* F = newA(edge, 2 * m);
  int mm  = sequence::filter(E, F, m, nEQF());
  //  parallel_for (int i=0; i < mm; i++) {
  parallel_doall(int, i, 0, mm) {
    F[i + mm].u = F[i].v;
    F[i + mm].v = F[i].u;
  }
  parallel_doall_end edgeArray R =
      remDuplicates(edgeArray(F, A.numRows, A.numCols, 2 * mm));
  free(F);
  return R;
}

struct getuF {
  vindex operator()(edge e) { return e.u; }
};

graph graphFromEdges(edgeArray EA, bool makeSym) {
  edgeArray A;
  if (makeSym)
    A = makeSymmetric(EA);
  else { // should have copy constructor
    edge* E = newA(edge, EA.nonZeros);
    //    parallel_for (int i=0; i < EA.nonZeros; i++) E[i] = EA.E[i];
    parallel_doall(int, i, 0, EA.nonZeros) { E[i] = EA.E[i]; }
    parallel_doall_end A = edgeArray(E, EA.numRows, EA.numCols, EA.nonZeros);
  }
  int m        = A.nonZeros;
  int n        = A.numRows;
  int* offsets = newA(int, n * 2);
  intSort::iSort(A.E, offsets, m, n, getuF());
  int* X    = newA(int, m);
  vertex* v = newA(vertex, n);
  //  parallel_for (int i=0; i < n; i++) {
  parallel_doall(int, i, 0, n) {
    int o          = offsets[i];
    int l          = ((i == n - 1) ? m : offsets[i + 1]) - offsets[i];
    v[i].degree    = l;
    v[i].Neighbors = X + o;
    for (int j = 0; j < l; j++) {
      v[i].Neighbors[j] = A.E[o + j].v;
    }
  }
  parallel_doall_end A.del();
  free(offsets);
  return graph(v, n, m, X);
}

edgeArray edgesFromGraph(graph G) {
  int numRows  = G.n;
  int nonZeros = G.m;
  vertex* V    = G.V;
  edge* E      = newA(edge, nonZeros);
  int k        = 0;
  for (int j = 0; j < numRows; j++)
    for (int i = 0; i < V[j].degree; i++)
      E[k++] = edge(j, V[j].Neighbors[i]);
  return edgeArray(E, numRows, numRows, nonZeros);
}

sparseRowMajorD sparseFromGraph(graph G) {
  int numRows  = G.n;
  int nonZeros = G.m;
  vertex* V    = G.V;
  int* Starts  = newA(int, numRows + 1);
  int* ColIds  = newA(int, nonZeros);
  int start    = 0;
  for (int i = 0; i < numRows; i++) {
    Starts[i] = start;
    start += V[i].degree;
  }
  Starts[numRows] = start;
  //  parallel_for (int j=0; j < numRows; j++)
  parallel_doall(int, j, 0, numRows) {
    for (int i = 0; i < (Starts[j + 1] - Starts[j]); i++) {
      ColIds[Starts[j] + i] = V[j].Neighbors[i];
    }
  }
  parallel_doall_end return sparseRowMajorD(numRows, numRows, nonZeros, Starts,
                                            ColIds, NULL);
}

// if I is NULL then it randomly reorders
graph graphReorder(graph Gr, int* I) {
  int n    = Gr.n;
  int m    = Gr.m;
  bool noI = (I == NULL);
  if (noI) {
    I = newA(int, Gr.n);
    //    parallel_for (int i=0; i < Gr.n; i++) I[i] = i;
    parallel_doall(int, i, 0, Gr.n) { I[i] = i; }
    parallel_doall_end random_shuffle(I, I + Gr.n);
  }
  vertex* V = newA(vertex, Gr.n);
  for (int i = 0; i < Gr.n; i++)
    V[I[i]] = Gr.V[i];
  for (int i = 0; i < Gr.n; i++) {
    for (int j = 0; j < V[i].degree; j++) {
      V[i].Neighbors[j] = I[V[i].Neighbors[j]];
    }
    sort(V[i].Neighbors, V[i].Neighbors + V[i].degree);
  }
  free(Gr.V);
  if (noI)
    free(I);
  return graph(V, n, m, Gr.allocatedInplace);
}

int graphCheckConsistency(graph Gr) {
  vertex* V     = Gr.V;
  int edgecount = 0;
  for (int i = 0; i < Gr.n; i++) {
    edgecount += V[i].degree;
    for (int j = 0; j < V[i].degree; j++) {
      vindex ngh = V[i].Neighbors[j];
      utils::myAssert(ngh >= 0 && ngh < Gr.n,
                      "graphCheckConsistency: bad edge");
    }
  }
  if (Gr.m != edgecount) {
    cout << "bad edge count in graphCheckConsistency: m = " << Gr.m
         << " sum of degrees = " << edgecount << endl;
    abort();
  }
  return 0;
}

sparseRowMajorD sparseFromCsrFile(const char* fname) {
  FILE* f = fopen(fname, "r");
  if (f == NULL) {
    cout << "Trying to open nonexistant file: " << fname << endl;
    abort();
  }

  int numRows;
  int numCols;
  int nonZeros;
  int nc = fread(&numRows, sizeof(int), 1, f);
  nc     = fread(&numCols, sizeof(int), 1, f);
  nc     = fread(&nonZeros, sizeof(int), 1, f);

  double* Values  = newA(double, nonZeros);
  int* ColIds     = newA(int, nonZeros);
  int* Starts     = newA(int, (1 + numRows));
  Starts[numRows] = nonZeros;

  size_t r;
  r = fread(Values, sizeof(double), nonZeros, f);
  r = fread(ColIds, sizeof(int), nonZeros, f);
  r = fread(Starts, sizeof(int), numRows, f);
  fclose(f);
  return sparseRowMajorD(numRows, numCols, nonZeros, Starts, ColIds, Values);
}

edgeArray edgesFromMtxFile(const char* fname) {
  ifstream file(fname, ios::in);
  char* line = newA(char, 1000);
  int i, j = 0;
  while (file.peek() == '%') {
    j++;
    file.getline(line, 1000);
  }
  int numRows, numCols, nonZeros;
  file >> numRows >> numCols >> nonZeros;
  // cout << j << "," << numRows << "," << numCols << "," << nonZeros << endl;
  edge* E = newA(edge, nonZeros);
  double toss;
  for (i = 0, j = 0; i < nonZeros; i++) {
    file >> E[j].u >> E[j].v >> toss;
    E[j].u--;
    E[j].v--;
    if (toss != 0.0)
      j++;
  }
  nonZeros = j;
  // cout << "nonzeros = " << nonZeros << endl;
  file.close();
  return edgeArray(E, numRows, numCols, nonZeros);
}
