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
#include "sequence.h"
using namespace std;
using namespace benchIO;

// The serial spanning tree used for checking against
pair<int*,int> st(edgeArray EA);

// This needs to be augmented with a proper check

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;

  commandLine P(argc,argv,"<inFile> <outfile>");
  pair<char*,char*> fnames = P.IOFileNames();
  char* iFile = fnames.first;
  char* oFile = fnames.second;

  edgeArray In = readEdgeArrayFromFile(iFile);
  _seq<int> Out = readIntArrayFromFile(oFile);

  //run serial ST
  pair<int*,int> serialST = st(In);
  if (Out.n != serialST.second){
    cout << "Wrong edge count: ST has " << serialST.second << " edges but algorithm returned " << Out.n << " edges\n";
    return (1);
  }

  //check if ST has cycles by running serial ST on it
  //and seeing if result changes
  bool* flags = newA(bool,In.nonZeros);
  cilk_for(int i=0;i<In.nonZeros;i++) flags[i] = 0;
  cilk_for(int i=0;i<Out.n;i++)
    flags[Out.A[i]] = 1;
  edge* E = newA(edge,In.nonZeros);
  int m = sequence::pack(In.E,E,flags,In.nonZeros);
  edgeArray EA(E,In.numRows,In.numCols,m); 

  pair<int*,int> check = st(EA);
  if (m != check.second){
    cout << "Result is not a spanning tree " << endl;
    return (1);
  }
  free(flags);
  free(serialST.first);
  free(check.first);
  In.del();
  EA.del();

  return 0;
}
