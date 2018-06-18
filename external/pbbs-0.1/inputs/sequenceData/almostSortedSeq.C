#include "sequenceData.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
using namespace dataGen;
using namespace benchIO;

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "[-r <swaps>] [-t {int,double}] <size> <outfile>");
  pair<int, char*> in = P.sizeAndFileName();
  elementType tp      = elementTypeFromString(P.getOptionValue("-t", "int"));
  int n               = in.first;
  char* fname         = in.second;
  int swaps           = P.getOptionIntValue("-r", floor(sqrt((float)n)));

  switch (tp) {
  case intT:
    return writeSequenceToFile(almostSorted<int>(0, n, swaps), n, fname);
  case doubleT:
    return writeSequenceToFile(almostSorted<double>(0, n, swaps), n, fname);
  default:
    cout << "genSeqRand: not a valid type" << endl;
  }
}
