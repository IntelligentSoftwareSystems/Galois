#include "Galois/Runtime/ParaMeter.h"

static const unsigned FNAME_SIZE = 256;
static char statsFileName[FNAME_SIZE];

static bool& firstRun() {
  static bool isFirst = true;
  return isFirst;
}

static void printHeader(FILE* out) {
   fprintf(out, "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE\n");
}

void Galois::Runtime::ParaMeterInit::init() {
  if (firstRun()) {
    firstRun() = false;

    time_t rawtime;
    struct tm* timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(statsFileName, FNAME_SIZE, "ParaMeter_Stats_%Y-%m-%d_%H:%M:%S.csv", timeinfo);

    FILE* statsFH = fopen(statsFileName, "w");
    printHeader(statsFH);
    fclose(statsFH);
  }
}

const char* Galois::Runtime::ParaMeterInit::getStatsFileName() {
  return statsFileName;
}
