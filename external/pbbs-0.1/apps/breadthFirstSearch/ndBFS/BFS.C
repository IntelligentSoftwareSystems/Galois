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

#include "utils.h"
#include "sequence.h"
#include "graph.h"
#include "parallel.h"
using namespace std;

// **************************************************************
//    Non-DETERMINISTIC BREADTH FIRST SEARCH
// **************************************************************

// **************************************************************
//    THE NON-DETERMINISTIC BSF
//    Updates the graph so that it is the BFS tree (i.e. the neighbors
//      in the new graph are the children in the bfs tree)
// **************************************************************

struct nonNegF {
  bool operator()(int a) { return (a >= 0); }
};

pair<int, int> BFS(vindex start, graph GA) {
  int numVertices      = GA.n;
  int numEdges         = GA.m;
  vertex* G            = GA.V;
  vindex* Frontier     = newA(vindex, numEdges);
  int* Visited         = newA(vindex, numVertices);
  vindex* FrontierNext = newA(vindex, numEdges);
  int* Counts          = newA(int, numVertices);
  //  {parallel_for(int i = 0; i < numVertices; i++) Visited[i] = 0;}
  {parallel_doall(int, i, 0, numVertices){Visited[i] = 0;
}
parallel_doall_end
}

Frontier[0]      = start;
int frontierSize = 1;
Visited[start]   = 1;

int totalVisited = 0;
int round        = 0;

while (frontierSize > 0) {
  round++;
  totalVisited += frontierSize;

  //    {parallel_for (int i=0; i < frontierSize; i++)
  {
    parallel_doall(int, i, 0, frontierSize) {
      Counts[i] = G[Frontier[i]].degree;
    }
    parallel_doall_end
  }
  int nr = sequence::scan(Counts, Counts, frontierSize, utils::addF<int>(), 0);

  // For each vertexB in the frontier try to "hook" unvisited neighbors.
  //    {parallel_for(int i = 0; i < frontierSize; i++) {
  {parallel_doall(int, i, 0, frontierSize){int k = 0;
  vindex v = Frontier[i];
  int o    = Counts[i];
  for (int j = 0; j < G[v].degree; j++) {
    vindex ngh = G[v].Neighbors[j];
    if (Visited[ngh] == 0 && utils::CAS(&Visited[ngh], 0, 1)) {
      FrontierNext[o + j] = G[v].Neighbors[k++] = ngh;
    } else
      FrontierNext[o + j] = -1;
  }
  G[v].degree = k;
}
parallel_doall_end
}

// Filter out the empty slots (marked with -1)
frontierSize = sequence::filter(FrontierNext, Frontier, nr, nonNegF());
}
free(FrontierNext);
free(Frontier);
free(Counts);
free(Visited);
return pair<int, int>(totalVisited, round);
}
