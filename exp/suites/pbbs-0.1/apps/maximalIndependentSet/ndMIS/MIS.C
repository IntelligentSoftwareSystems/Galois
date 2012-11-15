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
using namespace std;

// **************************************************************
//    MAXIMAL INDEPENDENT SET
// **************************************************************

void maxIndSetNonDeterministic(int n, vertex* G, char* Flags){
  int* V = newA(int,n);
  int* failures = newA(int,n);

//  parallel_for(int i=0;i<n;i++) V[i]=INT_MAX;
  parallel_doall(int, i, 0, n) { V[i]=INT_MAX; } parallel_doall_end
  parallel_doall(int, i, 0, n) { failures[i] = 0; } parallel_doall_end
  
//  parallel_for(int i=0;i<n;i++){
  parallel_doall(int, i, 0, n) {
    vindex v = i;
    while(1){
      //drop out if already in or out of MIS
      if(Flags[v]) break;
      //try to lock self and neighbors
      if(utils::CAS(&V[v], INT_MAX, 0)) {
	int k = 0;
	for(int j=0; j<G[v].degree; j++){
	  vindex ngh = G[v].Neighbors[j];
	  //if ngh is not in MIS or we successfully 
	  //acquire lock, increment k
	  if(Flags[ngh] || utils::CAS(&V[ngh], INT_MAX, 0)) k++;
	  else break;
	}
	if(k == G[v].degree){ 
	  //win on self and neighbors so fill flags
	  Flags[v] = 1;
	  for(int j=0;j<G[v].degree;j++){
	    vindex ngh = G[v].Neighbors[j]; 
	    if(Flags[ngh] != 2) Flags[ngh] = 2;
	  }
	} else { 
	  //lose so reset V values up to point
	  //where it lost
	  V[v] = INT_MAX;
	  for(int j = 0; j < k; j++){
	    vindex ngh = G[v].Neighbors[j];
	    if(Flags[ngh] != 2) V[ngh] = INT_MAX;
	  }
          failures[v]++;
	}
      } else {
        failures[v]++;
      }
    }
  } parallel_doall_end
  int nfailures = sequence::plusReduce(failures, n);
  cout << "failures = " << nfailures << "\n";
  free(V); free(failures);

}

void brokenCompiler(char* Flags, int n) {
//  parallel_for (int i=0; i < n; i++) Flags[i] = 0;
  parallel_doall(int, i, 0, n) { Flags[i] = 0; } parallel_doall_end
}

char* maximalIndependentSet(graph G) {
  int n = G.n;
  char* Flags = newA(char,n);
  brokenCompiler(Flags, n);
  maxIndSetNonDeterministic(n, G.V, Flags);
  return Flags;
}
