
#ifdef GALOIS_EXP
#include "Galois/Runtime/ParaMeter.h"

static const unsigned FNAME_SIZE = 256;
static char statsFileName[ FNAME_SIZE ];

static bool& firstRun () {
  static bool isFirst = true;
  return isFirst;
}

using GaloisRuntime::ParaMeter;

void ParaMeter::init () {
  if (firstRun ()) {
    firstRun () = false;

    time_t rawtime;
    struct tm* timeinfo;

    time (&rawtime);
    timeinfo = localtime (&rawtime);

    strftime (statsFileName, FNAME_SIZE, "ParaMeter_Stats_%Y-%m-%d_%H:%M:%S.csv", timeinfo);

    FILE* statsFH = fopen (statsFileName, "w");
    ParaMeter::printHeader (statsFH);
    fclose (statsFH);
  }
}

const char* ParaMeter::getStatsFileName () {
  return statsFileName;
}

#endif
