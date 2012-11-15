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

#include <math.h>
#include "IO.h"
#include "parseCommandLine.h"
#include "graph.h"
#include "graphIO.h"
#include "graphUtils.h"
#include "dataGen.h"
#include "parallel.h"
using namespace benchIO;
using namespace dataGen;
using namespace std;

int loc2d(int n, int i1, int i2) {
  return ((i1 + n) % n)*n + (i2 + n) % n;
}

edgeArray edge2DMesh(int n) {
  int dn = round(pow((double) n,1.0/2.0));
  int nn = dn*dn;
  int nonZeros = 2*nn;
  edge *E = newA(edge,nonZeros);
//  parallel_for (int i=0; i < dn; i++)
  parallel_doall(int, i, 0, dn) {
    for (int j=0; j < dn; j++) {
      int l = loc2d(dn,i,j);
      E[2*l] = edge(l,loc2d(dn,i+1,j));
      E[2*l+1] = edge(l,loc2d(dn,i,j+1));
    }
  } parallel_doall_end
  return edgeArray(E,nn,nn,nonZeros);
}

int loc3d(int n, int i1, int i2, int i3) {
  return ((i1 + n) % n)*n*n + ((i2 + n) % n)*n + (i3 + n) % n;
}

edgeArray edge3DMesh(int n) {
  int dn = round(pow((double) n,1.0/3.0));
  int nn = dn*dn*dn;
  int nonZeros = 3*nn;
  edge *E = newA(edge,nonZeros);
//  parallel_for (int i=0; i < dn; i++)
  parallel_doall(int, i, 0, dn) {
    for (int j=0; j < dn; j++) 
      for (int k=0; k < dn; k++) {
	int l = loc3d(dn,i,j,k);
	E[3*l] =   edge(l,loc3d(dn,i+1,j,k));
	E[3*l+1] = edge(l,loc3d(dn,i,j+1,k));
	E[3*l+2] = edge(l,loc3d(dn,i,j,k+1));
      }
  } parallel_doall_end
  return edgeArray(E,nn,nn,nonZeros);
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"[-d {2,3}] [-j] [-o] n <outFile>");
  pair<int,char*> in = P.sizeAndFileName();
  int n = in.first;
  char* fname = in.second;
  int dims = P.getOptionIntValue("-d", 2);
  bool ordered = P.getOption("-o");
  bool adjArray = P.getOption("-j");
  edgeArray EA;
  if (dims == 2) 
    EA = edge2DMesh(n);
  else if (dims == 3) 
    EA = edge3DMesh(n);
  else 
    P.badArgument();
  if (adjArray) {
    graph G = graphFromEdges(EA,1);
    if (!ordered) G = graphReorder(G, NULL);
    return writeGraphToFile(G, fname);
  } else {
    if (!ordered) std::random_shuffle(EA.E, EA.E + EA.nonZeros);
    return writeEdgeArrayToFile(EA, fname);
  }
}
