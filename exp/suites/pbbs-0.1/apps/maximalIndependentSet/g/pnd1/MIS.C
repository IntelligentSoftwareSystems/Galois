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
#include <deque>
#include "sequence.h"
#include "graph.h"
#include "parallel.h"
//#include "speculative_for.h"

struct TLD {
  std::deque<int> abortedQ;
  int aborted;
  TLD(): aborted(0) { }
};

template<class S>
struct GFn1 {
  S step;
  TLD* tld;
  GFn1(S _step, TLD* _tld): step(_step), tld(_tld) { }
  void operator()(int i) {
        unsigned tid = Exp::getTID();
        TLD& t = tld[tid];
        int cur = i;
        while (true) {
#ifdef DUMB
          if (cur >= numberKeep) I[cur] = numberDone + cur;
#endif
          bool success = step.commit(cur);
          //bool success = state[i].commit(I[i]);
          if (!success) {
            t.abortedQ.push_back(cur);
            t.aborted++;
          }
#ifdef DUMB
          keep[cur] = !success;
#endif
          if (!success)
            break;
          if (!t.abortedQ.empty()) {
            cur = t.abortedQ.front();
            t.abortedQ.pop_front();
          } else {
            break;
          }
        }
  }
};

//#define DUMB
template <class S>
void speculative_for(S step, int s, int e, int granularity, bool hasState=0, int maxTries=-1) {
  unsigned numThreads = Exp::getNumThreads();
  
  if (maxTries < 0) maxTries = 2*granularity;
  int maxRoundSize = (int) numThreads;
#ifdef DUMB
  maxRoundSize = (e-s)/granularity+1;
  vindex *I = newA(vindex,maxRoundSize);
  vindex *Ihold = newA(vindex,maxRoundSize);
  bool *keep = newA(bool,maxRoundSize);
  S *state;
  if (hasState) {
    state = newA(S, maxRoundSize);
    for (int i=0; i < maxRoundSize; i++) state[i] = step;
  }
#endif
  TLD *tld = new TLD[maxRoundSize];

  int round = 0; 
  int numberDone = s; // number of iterations done
  int numberKeep = 0; // number of iterations to carry to next round
  int failed = 0;

  while (numberDone < e) {
    //cout << "numberDone=" << numberDone << endl;
    if (round++ > maxTries) {
//      cerr << "speculativeLoop: too many iterations, increase maxTries parameter\n";
//      abort();
    }
    //int size = min(maxRoundSize, e - numberDone);
    int size = e - numberDone;

    if (!hasState) {
//      parallel_for (int i =0; i < size; i++) {
      GFn1<S> gfn1(step, tld);
      parallel_doall_obj(int, i, 0, size, gfn1)  {
        unsigned tid = Exp::getTID();
        TLD& t = tld[tid];
        int cur = i;
        while (true) {
#ifdef DUMB
          if (cur >= numberKeep) I[cur] = numberDone + cur;
#endif
          bool success = step.commit(cur);
          //bool success = state[i].commit(I[i]);
          if (!success) {
            t.abortedQ.push_back(cur);
            t.aborted++;
          }
#ifdef DUMB
          keep[cur] = !success;
#endif
          if (!success)
            break;
          if (!t.abortedQ.empty()) {
            cur = t.abortedQ.front();
            t.abortedQ.pop_front();
          } else {
            break;
          }
        }
      } parallel_doall_end

      for (int i = 0; i < numThreads; ++i) {
        TLD& t = tld[i];
        for (int j = 0; j < t.abortedQ.size(); ++j) {
          int cur = t.abortedQ[j];
          bool success = step.commit(cur);
          if (!success)
            abort();
        }
        failed += t.aborted;
        t.abortedQ.clear();
        t.aborted = 0;
      }
    } else {
      abort();
    }

#ifdef DUMB
    // keep edges that failed to hook for next round
    numberKeep = sequence::pack(I, Ihold, keep, size);
    failed += numberKeep;
    swap(I, Ihold);
    numberDone += size - numberKeep;
#endif
    numberKeep = 0;
    numberDone += size - numberKeep;
  }
#ifdef DUMB
  free(I); free(Ihold); free(keep); free(state);
#endif
  delete [] tld;
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
  char *Flags;  vertex*G; int *Marks;
  pthread_mutex_t* locks;
  MISstep(char* _F, vertex* _G, int* _M, pthread_mutex_t* _l) : Flags(_F), G(_G), Marks(_M), locks(_l) {}

  bool acquire(int id, int i) {
    if (Marks[i] == id)
      return true;

    bool retval;
    pthread_mutex_lock(&locks[i]);
    int v = Marks[i];
    if (v == -1) {
      Marks[i] = id;
      retval = true;
    } else {
      retval = false;
    }
    pthread_mutex_unlock(&locks[i]);
    return retval;
  }

  bool release(int id, int i) {
    if (Marks[i] != id)
      return false;

    Marks[i] = -1;
    return true;
  }

  bool doit(int i) {
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
    bool retval = doit(i);
    resetState(i);
    return retval;
  }
};

char* maximalIndependentSet(graph GS) {
  int n = GS.n;
  vertex* G = GS.V;
  int* Marks = newArray(n, -1);
  char* Flags = newArray(n,  (char) 0);
  pthread_mutex_t* locks = new pthread_mutex_t[n];
  for (int i = 0; i < n; ++i)
    pthread_mutex_init(&locks[i], NULL);
  MISstep mis(Flags, G, Marks, locks);
  int numRounds = Exp::getNumRounds();
  numRounds = numRounds <= 0 ? 25 : numRounds;
  speculative_for(mis, 0, n, numRounds);
  for (int i = 0; i < n; ++i)
    pthread_mutex_destroy(&locks[i]);
  delete [] locks;
  free(Marks);
  return Flags;
}
