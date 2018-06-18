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

template <class S>
struct GFn1 {
  int numberDone;
  S step;
  GFn1(int _numberDone, S _step) : numberDone(_numberDone), step(_step) {}
  void operator()(int i) {
    int II = numberDone + i;
    step.reserve(II);
  }
};

template <class S>
struct GFn2 {
  int numberDone;
  S step;
  GFn2(int _numberDone, S _step) : numberDone(_numberDone), step(_step) {}
  void operator()(int i) {
    int II = numberDone + i;
    step.commit(II);
  }
};

template <class S>
void speculative_for(S step, int s, int e, int granularity, bool hasState = 1,
                     int maxTries = -1) {
  if (maxTries < 0)
    maxTries = 2 * granularity;
  int maxRoundSize = (e - s) / granularity + 1;
  S* state;
  if (hasState) {
    state = newA(S, maxRoundSize);
    for (int i = 0; i < maxRoundSize; i++)
      state[i] = step;
  }

  int round      = 0;
  int numberDone = s; // number of iterations done
  int numberKeep = 0; // number of iterations to carry to next round
  int failed     = 0;

  while (numberDone < e) {
    // cout << "numberDone=" << numberDone << endl;
    if (round++ > maxTries) {
      //      cerr << "speculativeLoop: too many iterations, increase maxTries
      //      parameter\n"; abort();
    }
    int size = min(maxRoundSize, e - numberDone);

    if (hasState) {
      abort();
    } else {
      //      parallel_for (int i =0; i < size; i++) {
      GFn1<S> gfn1(numberDone, step);
      parallel_doall_obj(int, i, 0, size, gfn1) {
        int II = numberDone + i;
        step.reserve(II);
      }
      parallel_doall_end
    }

    if (hasState) {
      abort();
    } else {
      //      parallel_for (int i =0; i < size; i++) {
      GFn2<S> gfn2(numberDone, step);
      parallel_doall_obj(int, i, 0, size, gfn2) {
        int II = numberDone + i;
        step.commit(II);
      }
      parallel_doall_end
    }

    // keep edges that failed to hook for next round
    numberKeep = 0;
    failed += numberKeep;
    numberDone += size - numberKeep;
  }
  if (hasState)
    free(state);
  // cout << "rounds = " << round << " failed = " << failed << "\n";
}

struct BFSstep {
  vertex* G;
  vindex* Frontier;
  int* Visited;
  vindex* FrontierNext;
  int* Counts;
  int* Marks;
  pthread_mutex_t* locks;

  BFSstep(vertex* _G, vindex* _Frontier, int* _Visited, vindex* _FrontierNext,
          int* _Counts, int* _Marks, pthread_mutex_t* _locks)
      : G(_G), Frontier(_Frontier), Visited(_Visited),
        FrontierNext(_FrontierNext), Counts(_Counts), Marks(_Marks),
        locks(_locks) {}

  void writeMin(vindex ngh, vindex v) {
    pthread_mutex_lock(&locks[ngh]);
    if (Visited[ngh] > v) {
      Visited[ngh] = v;
    }
    pthread_mutex_unlock(&locks[ngh]);
  }

  bool reserve(int i) {
    int k    = 0;
    vindex v = Frontier[i];
    int o    = Counts[i];
    for (int j = 0; j < G[v].degree; j++) {
      vindex ngh = G[v].Neighbors[j];
      if (Visited[ngh] == -1)
        utils::writeMin(&Marks[ngh], v);
    }
    return true;
  }

  bool commit(int i) {
    int k    = 0;
    vindex v = Frontier[i];
    int o    = Counts[i];
    for (int j = 0; j < G[v].degree; j++) {
      vindex ngh = G[v].Neighbors[j];
      if (Marks[ngh] == v) {
        Visited[ngh]        = 1;
        FrontierNext[o + j] = G[v].Neighbors[k++] = ngh;
        Marks[ngh]                                = INT_MAX;
      } else
        FrontierNext[o + j] = -1;
    }
    G[v].degree = k;
    return true;
  }
};

pair<int, int> BFS(vindex start, graph GA) {
  int numRounds = Exp::getNumRounds();
  numRounds     = numRounds <= 0 ? 1 : numRounds;

  int numVertices      = GA.n;
  int numEdges         = GA.m;
  vertex* G            = GA.V;
  vindex* Frontier     = newA(vindex, numEdges);
  int* Visited         = newA(vindex, numVertices);
  int* Marks           = newA(vindex, numVertices);
  vindex* FrontierNext = newA(vindex, numEdges);
  int* Counts          = newA(int, numVertices);

  pthread_mutex_t* locks = new pthread_mutex_t[numVertices];
  for (int i = 0; i < numVertices; ++i)
    pthread_mutex_init(&locks[i], NULL);

  //  {parallel_for(int i = 0; i < numVertices; i++) Visited[i] = 0;}
  {parallel_doall(int, i, 0, numVertices){Visited[i] = -1;
}
parallel_doall_end
}
{parallel_doall(int, i, 0, numVertices){Marks[i] = INT_MAX;
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

  BFSstep bfs(G, Frontier, Visited, FrontierNext, Counts, Marks, locks);
  speculative_for(bfs, 0, frontierSize, numRounds, 0);

  // Filter out the empty slots (marked with -1)
  frontierSize = sequence::filter(FrontierNext, Frontier, nr, nonNegF());
}

for (int i = 0; i < numVertices; ++i)
  pthread_mutex_destroy(&locks[i]);

free(FrontierNext);
free(Frontier);
free(Counts);
free(Visited);
free(Marks);
return pair<int, int>(totalVisited, round);
}
