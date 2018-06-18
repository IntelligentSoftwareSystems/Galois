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
#include "sequence.h"
#include "graph.h"
#include "parallel.h"
//#include "speculative_for.h"
template <class S>
void speculative_for(S step, int s, int e, int granularity, bool hasState = 1,
                     int maxTries = -1) {
  if (maxTries < 0)
    maxTries = 2 * granularity;
  int maxRoundSize = (e - s) / granularity + 1;
  vindex* I        = newA(vindex, maxRoundSize);
  vindex* Ihold    = newA(vindex, maxRoundSize);
  bool* keep       = newA(bool, maxRoundSize);
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
      parallel_doall(int, i, 0, size) {
        if (i >= numberKeep)
          I[i] = numberDone + i;
        keep[i] = step.reserve(I[i]);
      }
      parallel_doall_end
    }

    if (hasState) {
      abort();
    } else {
      //      parallel_for (int i =0; i < size; i++) {
      parallel_doall(int, i, 0, size) {
        if (keep[i])
          keep[i] = !step.commit(I[i]);
      }
      parallel_doall_end
    }

    // keep edges that failed to hook for next round
    numberKeep = sequence::pack(I, Ihold, keep, size);
    failed += numberKeep;
    swap(I, Ihold);
    numberDone += size - numberKeep;
  }
  free(I);
  free(Ihold);
  free(keep);
  if (hasState)
    free(state);
  cout << "rounds = " << round << " failed = " << failed << "\n";
}

using namespace std;

// **************************************************************
//    MAXIMAL INDEPENDENT SET
// **************************************************************

// For each vertex:
//   Flags = 0 indicates undecided
//   Flags = 1 indicates chosen
//   Flags = 2 indicates a neighbor is chosen
struct MISstep {
  char* Flags;
  vertex* G;
  int* Marks;
  MISstep(char* _F, vertex* _G, int* _M) : Flags(_F), G(_G), Marks(_M) {}

  bool acquire(int id, int i) {
    int v;
    do {
      v = Marks[i];
      if (v == id)
        return true;
      if (v == -1 && __sync_bool_compare_and_swap(&Marks[i], v, id)) {
        return true;
      }
    } while (v == -1);

    return false;
  }

  bool release(int id, int i) {
    if (Marks[i] != id)
      return false;

    Marks[i] = -1;
    return true;
  }

  bool reserve(int i) {
    if (doit(i, true))
      return true;
    resetState(i);
    return false;
  }

  bool doit(int i, bool reserve) {
    int d = G[i].degree;
    if (!acquire(i, i))
      return false;
    if (Flags[i] != 0)
      return true;
    for (int j = 0; j < d; j++) {
      vindex ngh = G[i].Neighbors[j];
      if (!acquire(i, ngh))
        return false;
      if (Flags[ngh] != 0)
        return true;
    }
    if (reserve)
      return true;
    Flags[i] = 1;
    for (int j = 0; j < d; j++) {
      vindex ngh = G[i].Neighbors[j];
      Flags[ngh] = 2;
    }
    return true;
  }

  void resetState(int i) {
    int d = G[i].degree;
    if (!release(i, i))
      return;
    for (int j = 0; j < d; j++) {
      vindex ngh = G[i].Neighbors[j];
      if (!release(i, ngh))
        return;
    }
    return;
  }

  bool commit(int i) {
    bool retval = doit(i, false);
    resetState(i);
    return retval;
  }
};

char* maximalIndependentSet(graph GS) {
  int n       = GS.n;
  vertex* G   = GS.V;
  int* Marks  = newArray(n, -1);
  char* Flags = newArray(n, (char)0);
  MISstep mis(Flags, G, Marks);
  int numRounds = Exp::getNumRounds();
  numRounds     = numRounds <= 0 ? 25 : numRounds;
  speculative_for(mis, 0, n, numRounds, 0);
  free(Marks);
  return Flags;
}
