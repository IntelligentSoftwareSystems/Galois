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
#include "IO.h"
#include "graph.h"
#include "graphIO.h"
#include "parseCommandLine.h"
using namespace std;
using namespace benchIO;

int levelNumber(int start, int level, vindex* P, vindex* L, graph T) {
  if (L[start] != -1) {
    cout << "BFSCheck: not a tree" << endl;
    return 1;
  }
  L[start] = level;
  for (int i = 0; i < T.V[start].degree; i++) {
    vindex n = T.V[start].Neighbors[i];
    P[n]     = start;
    levelNumber(n, level + 1, P, L, T);
  }
  return 0;
}

// Checks if T is valid BFS tree relative to G starting at i
int checkBFS(int start, graph G, graph T) {
  if (G.n != T.n) {
    cout << "BFSCheck: vertex counts don't match: " << G.n << ", " << T.n
         << endl;
    return 1;
  }
  if (T.m > G.n - 1) {
    cout << "BFSCheck: too many edges in tree " << endl;
    return 1;
  }
  vindex* P = newA(vindex, G.n);
  vindex* L = newA(vindex, G.n);
  //  parallel_for (int i=0; i < G.n; i++) {
  parallel_doall(int, i, 0, G.n) {
    P[i] = -1;
    L[i] = -1;
  }
  parallel_doall_end if (levelNumber(start, 0, P, L, T)) return 1;
  for (int i = 0; i < G.n; i++) {
    bool Check = 0;
    if (L[i] == -1) {
      for (int j = 0; j < G.V[i].degree; j++) {
        vindex ngh = G.V[i].Neighbors[j];
        if (L[ngh] != -1) {
          cout << "BFSCheck: connected vertex not in tree " << endl;
          return 1;
        }
      }
    } else {
      for (int j = 0; j < G.V[i].degree; j++) {
        vindex ngh = G.V[i].Neighbors[j];
        if (P[i] == ngh)
          Check = 1;
        else if (L[ngh] > L[i] + 1 || L[ngh] < L[i] - 1) {
          cout << "BFSCheck: edge spans two levels " << endl;
          return 1;
        }
      }
      if (i != start && Check == 0) {
        cout << "BFSCheck: parent not an edge " << endl;
        return 1;
      }
    }
  }
  return 0;
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "<inFile> <outfile>");
  pair<char*, char*> fnames = P.IOFileNames();
  char* iFile               = fnames.first;
  char* oFile               = fnames.second;

  graph G = readGraphFromFile(iFile);
  graph T = readGraphFromFile(oFile);

  return checkBFS(0, G, T);
}
