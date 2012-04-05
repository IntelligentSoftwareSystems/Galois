#include "parallel.h"
#include "sequenceIO.h"
#include "parseCommandLine.h"
using namespace benchIO;

char* trigramString(int s, int e);

int parallel_main(int argc, char* argv[]) {
  commandLine P(argc,argv,"<size> <outfile>");
  pair<int,char*> in = P.sizeAndFileName();
  char *A = trigramString(0,in.first);
  return writeStringToFile(A, in.first, in.second);
}
