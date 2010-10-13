#include "Galois/Launcher.h"


// This is linux/bsd specific
#include <sys/time.h>

static bool firstRun = true;
static timeval start;
static timeval stop;

bool Galois::Launcher::isFirstRun() {
  return firstRun;
}

void Galois::Launcher::startTiming() {
  gettimeofday(&start, 0);
}

void Galois::Launcher::stopTiming() {
  gettimeofday(&stop, 0);
}

void Galois::Launcher::reset() {
  firstRun = false;
}

unsigned long Galois::Launcher::elapsedTime() {
  unsigned long msec = stop.tv_sec - start.tv_sec;
  msec *= 1000;
  if (stop.tv_usec > start.tv_usec)
    msec += (stop.tv_usec - start.tv_usec) / 1000;
  else {
    msec -= 1; //borrow
    msec += (stop.tv_usec + 1000 - start.tv_usec) / 1000;
  }
  return msec;
  
}
