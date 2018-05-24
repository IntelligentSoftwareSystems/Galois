#include "galois/runtime/Executor_ParaMeter.h"
#include "galois/gIO.h"
#include "galois/substrate/EnvCheck.h"



#include <ctime>

struct StatsFileManager {

  constexpr static const char* const PARAM_FILE_ENV_VAR = "GALOIS_PARAMETER_OUTFILE";


  bool init = false;
  bool isOpen = false;
  FILE* statsFH = nullptr;
  // char statsFileName[FNAME_SIZE];
  std::string statsFileName;


  ~StatsFileManager (void) {
    close ();
  }

  static void getTimeStampedName(std::string& statsFileName) {

    constexpr unsigned FNAME_SIZE = 256;
    char buf[FNAME_SIZE];

    time_t rawtime;
    struct tm* timeinfo;
    
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    
    strftime(buf, FNAME_SIZE, "ParaMeter-Stats-%Y-%m-%d--%H-%M-%S.csv", timeinfo);
    statsFileName = buf;
  }

  FILE* get (void) {
    if (!init) {
      init = true;

      if (!galois::substrate::EnvCheck(PARAM_FILE_ENV_VAR, statsFileName)) {
        // statsFileName = "ParaMeter-Stats.csv";
        getTimeStampedName(statsFileName);
      }

      statsFH = fopen(statsFileName.c_str(), "w");
      GALOIS_ASSERT (statsFH != nullptr, "ParaMeter stats file error");

      galois::runtime::ParaMeter::StepStatsBase::printHeader(statsFH);

      fclose(statsFH);
    }

    if (!isOpen) {
      statsFH = fopen (statsFileName.c_str(), "a"); // open in append mode
      GALOIS_ASSERT (statsFH != nullptr, "ParaMeter stats file error");

      isOpen = true;
    }

    return statsFH;

  }

  void close (void) {
    if (isOpen) {
      fclose (statsFH);
      isOpen = false;
      statsFH = nullptr;
    }
  }

};

static StatsFileManager& getStatsFileManager (void) {
  static StatsFileManager s;
  return s;
}

FILE* galois::runtime::ParaMeter::getStatsFile (void) {
  return getStatsFileManager ().get ();
}

void galois::runtime::ParaMeter::closeStatsFile (void) {
  getStatsFileManager ().close ();
}

