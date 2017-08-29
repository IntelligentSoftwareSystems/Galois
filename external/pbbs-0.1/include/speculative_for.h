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

#include "parallel.h"
#include "utils.h"
#include "sequence.h"

struct reservation {
  int r;
  reservation() : r(INT_MAX) {}
  void reserve(int i) { utils::writeMin(&r, i); }
  bool reserved() { return (r < INT_MAX);}
  void reset() {r = INT_MAX;}
  bool check(int i) { return (r == i);}
  bool checkReset(int i) {
    if (r==i) { r = INT_MAX; return 1;}
    else return 0;
  }
};

inline void reserveLoc(int& x, int i) {utils::writeMin(&x,i);}

template <class S>
void speculative_for(S step, int s, int e, int granularity, 
		     bool hasState=1, int maxTries=-1) {
  if (maxTries < 0) maxTries = 2*granularity;
  int maxRoundSize = (e-s)/granularity+1;
  vindex *I = newA(vindex,maxRoundSize);
  vindex *Ihold = newA(vindex,maxRoundSize);
  bool *keep = newA(bool,maxRoundSize);
  S *state;
  if (hasState) {
    state = newA(S, maxRoundSize);
    for (int i=0; i < maxRoundSize; i++) state[i] = step;
  }

  int round = 0; 
  int numberDone = s; // number of iterations done
  int numberKeep = 0; // number of iterations to carry to next round

  while (numberDone < e) {
    //cout << "numberDone=" << numberDone << endl;
    if (round++ > maxTries) {
//      "speculativeLoop: too many iterations, increase maxTries parameter";
//      abort();
    }
    int size = min(maxRoundSize, e - numberDone);

    if (hasState) {
//      parallel_for (int i =0; i < size; i++) {
      parallel_doall(int, i, 0, size)  {
	if (i >= numberKeep) I[i] = numberDone + i;
	keep[i] = state[i].reserve(I[i]);
      } parallel_doall_end
    } else {
//      parallel_for (int i =0; i < size; i++) {
      parallel_doall(int, i, 0, size)  {
	if (i >= numberKeep) I[i] = numberDone + i;
	keep[i] = step.reserve(I[i]);
      } parallel_doall_end
    }

    if (hasState) {
//      parallel_for (int i =0; i < size; i++) 
      parallel_doall(int, i, 0, size) {
	if (keep[i]) keep[i] = !state[i].commit(I[i]);
      } parallel_doall_end
    } else {
//      parallel_for (int i =0; i < size; i++) 
      parallel_doall(int, i, 0, size) {
	if (keep[i]) keep[i] = !step.commit(I[i]);
      } parallel_doall_end
    }

    // keep edges that failed to hook for next round
    numberKeep = sequence::pack(I, Ihold, keep, size);
    swap(I, Ihold);
    numberDone += size - numberKeep;
  }
  free(I); free(Ihold); free(keep); free(state);
}
