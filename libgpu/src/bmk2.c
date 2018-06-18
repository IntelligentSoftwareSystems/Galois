#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmk2.h"

static int inited = 0;
static int bmk2   = 0;
static char* binid;
static char* inputid;
static char* runid;

char* bmk2_get_binid() { return binid; }

char* bmk2_get_inputid() { return inputid; }

char* bmk2_get_runid() { return runid; }

int bmk2_log_collect(const char* component, const char* file) {
  if (bmk2 && binid && inputid && runid) {
    fprintf(stderr, "COLLECT %s/%s %s %s %s\n", binid, inputid, runid,
            component, file);
    return 1;
  }

  return 0;
}

__attribute__((constructor)) void init_bmk2() {
  char* p;

  inited = 1;

  if (p = getenv("BMK2")) {
    if (atoi(p) == 1) {
      bmk2 = 1;
    }
  }

  if (bmk2) {
    if (p = getenv("BMK2_BINID")) {
      binid = strdup(p);
    }

    if (p = getenv("BMK2_INPUTID")) {
      inputid = strdup(p);
    }

    if (p = getenv("BMK2_RUNID")) {
      runid = strdup(p);
    }
  }
}
