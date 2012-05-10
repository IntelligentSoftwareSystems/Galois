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

#include <iostream>
#include "parallel.h"
#include "utils.h"
#include "sequence.h"
using namespace std;

template <class E>
void randPerm(E *A, int n) {
  int *I = newA(int,n);
  int *H = newA(int,n);
  int *check = newA(int,n);
  if (n < 100000) {
    for (int i=n-1; i > 0; i--) 
      swap(A[utils::hash(i)%(i+1)],A[i]);
    return;
  }

//  parallel_for (int i=0; i < n; i++) {
  parallel_doall(int, i, 0, n)  {
    H[i] = utils::hash(i)%(i+1);
    I[i] = i;
    check[i] = i;
  } parallel_doall_end

  int end = n;
  int ratio = 100;
  int maxR = 1 + n/ratio;
  int wasted = 0;
  int round = 0;
  int *hold = newA(int,maxR);
  bool *flags = newA(bool,maxR);
  //int *H = newA(int,maxR);

  while (end > 0) {
    round++;
    //if (round > 10 * ratio) abort();
    int size = 1 + end/ratio;
    int start = end-size;

//    {parallel_for(int i = 0; i < size; i++) {
    {parallel_doall(int, i, 0, size)  {
      int idx = I[i+start];
      int idy = H[idx];
      utils::writeMax(&check[idy], idx);
      } parallel_doall_end
    }

//    {parallel_for(int i = 0; i < size; i++) {
    {parallel_doall(int, i, 0, size)  {
      int idx = I[i+start];
      int idy = H[idx];
      flags[i] = 1;
      hold[i] = idx;
      if (check[idy] == idx ) {
	if (check[idx] == idx) {
	  swap(A[idx],A[idy]);
	  flags[i] = 0;
	}
	check[idy] = idy;
      }
      } parallel_doall_end
    }
    int nn = sequence::pack(hold,I+start,flags,size);
    end = end - size + nn;
    wasted += nn;
  }
  free(H); free(I); free(check); free(hold); free(flags);
  //cout << "wasted = " << wasted << " rounds = " << round  << endl;
}
