#include "parallel.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
using namespace benchIO;

char** trigramWords(int s, int e);

int parallel_main(int argc, char* argv[]) {
  commandLine P(argc,argv,"<size> <outfile>");
  pair<int,char*> in = P.sizeAndFileName();
  char **A = trigramWords(0,in.first);
  return writeSequenceToFile(A,in.first,in.second);
}
