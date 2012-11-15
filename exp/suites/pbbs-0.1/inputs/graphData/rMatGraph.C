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

#include "IO.h"
#include "parseCommandLine.h"
#include "graph.h"
#include "graphIO.h"
#include "dataGen.h"
#include "graphUtils.h"
#include "parallel.h"
using namespace benchIO;
using namespace dataGen;
using namespace std;

struct rMat {
  double a, ab, abc;
  int n; 
  unsigned int h;
  rMat(int _n, unsigned int _seed, 
       double _a, double _b, double _c) {
    n = _n; a = _a; ab = _a + _b; abc = _a+_b+_c;
    h = dataGen::hash<unsigned int>(_seed);
    utils::myAssert(abc <= 1.0,
		    "in rMat: a + b + c add to more than 1");
    utils::myAssert((1 << utils::log2Up(n)) == n, 
		    "in rMat: n not a power of 2");
  }

  edge rMatRec(int nn, int randStart, int randStride) {
    if (nn==1) return edge(0,0);
    else {
      edge x = rMatRec(nn/2, randStart + randStride, randStride);
      double r = dataGen::hash<double>(randStart);
      if (r < a) return x;
      else if (r < ab) return edge(x.u,x.v+nn/2);
      else if (r < abc) return edge(x.u+nn/2, x.v);
      else return edge(x.u+nn/2, x.v+nn/2);
    }
  }

  edge operator() (int i) {
    unsigned int randStart = dataGen::hash<unsigned int>((2*i)*h);
    unsigned int randStride = dataGen::hash<unsigned int>((2*i+1)*h);
    return rMatRec(n, randStart, randStride);
  }
};

edgeArray edgeRmat(int n, int m, unsigned int seed, 
		   float a, float b, float c) {
  int nn = (1 << utils::log2Up(n));
  rMat g(nn,seed,a,b,c);
  edge* E = newA(edge,m);
//  parallel_for (int i = 0; i < m; i++) 
  parallel_doall(int, i, 0, m) {
    E[i] = g(i);
  } parallel_doall_end
  return edgeArray(E,nn,nn,m);
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,
		"[-m <numedges>] [-s <intseed>] [-o] [-j] [-a <a>] [-b <b>] [-c <c>] n <outFile>");
  pair<int,char*> in = P.sizeAndFileName();
  int n = in.first;
  char* fname = in.second;
  double a = P.getOptionDoubleValue("-a",.5);
  double b = P.getOptionDoubleValue("-b",.1);
  double c = P.getOptionDoubleValue("-c", b);
  int m = P.getOptionIntValue("-m", 10*n);
  int seed = P.getOptionIntValue("-s", 1);
  bool adjArray = P.getOption("-j");
  bool ordered = P.getOption("-o");

  edgeArray EA = edgeRmat(n, m, seed, a, b, c);
  if (!ordered) {
    graph G = graphFromEdges(EA,adjArray);
    EA.del();
    G = graphReorder(G, NULL);
    if (adjArray) {
      writeGraphToFile(G, fname);
    } else {
      EA = edgesFromGraph(G);
      std::random_shuffle(EA.E, EA.E + m);
      writeEdgeArrayToFile(EA, fname);
    }
  } else {
    if (adjArray) {
      graph G = graphFromEdges(EA, 1);
      writeGraphToFile(G, fname);
    } else {
      writeEdgeArrayToFile(EA, fname);
    }
  }
}
