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

pair<int*,int> mst(wghEdgeArray G);

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"<inFile> <outfile>");
  pair<char*,char*> fnames = P.IOFileNames();
  char* iFile = fnames.first;
  char* oFile = fnames.second;

  wghEdgeArray In = readWghEdgeArrayFromFile(iFile);
  _seq<int> Out = readIntArrayFromFile(oFile);

  //check num edges
  pair<int*,int> serialMST = mst(In);
  if (Out.n != serialMST.second){
    cout << "Wrong edge count: MST has " << serialMST.second << " edges but algorithm returned " << Out.n << " edges\n";
    return (1);
    }

  //check for cycles
  bool* flags = newA(bool,In.m);
  cilk_for(int i=0;i<In.m;i++) flags[i] = 0;
  cilk_for(int i=0;i<Out.n;i++)
    flags[Out.A[i]] = 1;

  wghEdge* E = newA(wghEdge,In.m);
  int m = sequence::pack(In.E,E,flags,In.m);
  wghEdgeArray EA(E,In.n,m); 

  pair<int*,int> check = mst(EA);
  if (m != check.second){
    cout << "Result is not a spanning tree " << endl;
    return (1);
  }

  //check weights
  //weight from file
  double* weights = newA(double,m);
  cilk_for(int i=0;i<m;i++) weights[i] = E[i].weight;
  double total = sequence::plusScan(weights,weights,m);
  
  //correct weight
  cilk_for(int i=0;i<In.m;i++) flags[i] = 0;
  cilk_for(int i=0;i<Out.n;i++)
    flags[serialMST.first[i]] = 1;
  m = sequence::pack(In.E,E,flags,In.m);
  cilk_for(int i=0;i<m;i++) weights[i] = E[i].weight;
  double correctTotal = sequence::plusScan(weights,weights,m);

  if(total != correctTotal) {
    cout << "MST has a weight of " << total << " but should have a weight of " << correctTotal << endl;
    return (1);
  }

  free(weights);
  free(flags);
  free(serialMST.first);
  free(check.first);
  In.del();
  EA.del();

  return 0;
}
