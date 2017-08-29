#include "sequenceData.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
using namespace dataGen;
using namespace benchIO;

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc,argv,"[-t {int,double}] <size> <outfile>");
  pair<int,char*> in = P.sizeAndFileName();
  elementType tp = elementTypeFromString(P.getOptionValue("-t","int"));
  switch(tp) {
  case intT: 
    return writeSequenceToFile(expDist<int>(0,in.first),in.first,in.second);
  case doubleT: 
    return writeSequenceToFile(expDist<double>(0,in.first),in.first,in.second);
  default:
    cout << "genSeqRand: not a valid type" << endl;
  }
}
