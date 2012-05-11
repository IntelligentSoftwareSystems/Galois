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

#include "sequenceData.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
using namespace dataGen;
using namespace benchIO;

template <class T1>
pair<T1,int>* addIntData(T1* A, int n, int range) {
  pair<T1,int>* R = new pair<T1,int>[n];
  for (int ii=0; ii < n; ii++) {
    R[ii].first = A[ii];
    R[ii].second = dataGen::hash<int>(n+ii) % range;
  }
  return R;
}

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv, "[-r <range>] [-t {int,double}] <inFile> <outFile>");
  pair<char*,char*> fnames = P.IOFileNames();
  char* ifile = fnames.first;
  char* ofile = fnames.second;
  seqData D = readSequenceFromFile(ifile);
  elementType dt = D.dt;
  int n = D.n;
  elementType dataDT = elementTypeFromString(P.getOptionValue("-t","int"));
  if (dataDT == none) dataDT = dt;

  char* rangeString = P.getOptionValue("-r");
  int range = n;
  if (rangeString != NULL) range = atoi(rangeString);

  switch(dt) {
  case intT: 
    switch (dataDT) {
    case intT: 
      return writeSequenceToFile(addIntData((int*) D.A, n, range),n,ofile);
    default:
      cout << "addData: not a valid type" << endl;
    }
  case stringT: 
    switch (dataDT) {
    case intT: 
      return writeSequenceToFile(addIntData((char**) D.A, n, range),n,ofile);
    default:
      cout << "addData: not a valid type" << endl;
    }
  default:
    cout << "addData: not a valid type" << endl;
  }
}
