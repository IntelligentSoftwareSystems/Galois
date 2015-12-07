/*
 * To use:
 *     #define PAPI
 *     #include "simplepapi.h"
 *
 * To link against PAPI in the Galois CMakeLists:
 *     include_directories(SYSTEM "/usr/lib64/papi-5.1.1/usr/include")
 *     link_directories("/usr/lib64/papi-5.1.1/usr/lib")
 *     app(UpCholesky UpCholesky.cpp EXTLIBS papi)
 *     app(UpCholeskySimpleGraph UpCholeskySimpleGraph.cpp EXTLIBS papi)
 */
#ifdef PAPI
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <papi.h>

// PAPI
static int EventSet=PAPI_NULL;
static int nevents=0;

static const char *default_events = "PAPI_FP_OPS,PAPI_TOT_INS,PAPI_BR_INS";
// Other useful combinations:
//
// FPU instruction analysis (galois.ices.utexas.edu only):
//   PAPI_FDV_INS,PAPI_FSQ_INS,PAPI_FAD_INS,PAPI_FML_INS
//
// Memory accesses/cache misses:
//   PAPI_LD_INS,PAPI_SR_INS,PAPI_L1_DCM,PAPI_L2_TCM
//   PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_DCM
//   PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_TCM

static void papi_warn(const char *id, int retval) {
  printf("%s: PAPI error %d: %s\n", id, retval, PAPI_strerror(retval));
}
static void papi_error(const char *id, int retval) {
  papi_warn(id, retval);
  exit(1);
}

void papi_start() {
  int retval, i;
  if ( EventSet == PAPI_NULL ) {
    /* Initialize the PAPI library */
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
      fprintf(stderr, "PAPI library init error!\n");
      exit(1);
    }
    printf("Initialized PAPI version %d.%d.%d\n", PAPI_VERSION_MAJOR(retval),
           PAPI_VERSION_MINOR(retval), PAPI_VERSION_REVISION(retval));

    /* Create the Event Set */
    if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
      papi_error("PAPI_create_eventset", retval);

    /* Load PAPI_EVENTS environment variable */
    const char *envstr = getenv("PAPI_EVENTS");
    if ( !envstr ) {
      printf("PAPI_EVENTS not set, using default event list.\n");
      envstr = default_events;
    }
    char *eventlist = strdup(envstr);
    if ( !eventlist )
      abort();
    for ( i=0; eventlist[i]; i++ ) {
      eventlist[i] = toupper(eventlist[i]);
    }

    /* Parse events list, add counters to the Event Set */
    char *eventname = NULL, *saveptr = NULL;
    while ( (eventname = strtok_r(saveptr ? NULL : eventlist, ", ",
                                  &saveptr)) ) {
      int event = 0;
      if ( (retval = PAPI_event_name_to_code(eventname, &event)) != PAPI_OK ) {
        char msg[256];
        snprintf(msg, 255, "PAPI_event_name_to_code(%s)", eventname);
        papi_warn(msg, retval);
        continue;
      }
      if ( (retval = PAPI_add_event(EventSet, event)) != PAPI_OK ) {
        char msg[256];
        snprintf(msg, 255, "PAPI_add_event(%s)", eventname);
        papi_warn(msg, retval);
        continue;
      }
      nevents++;
    }

    free(eventlist);
  }

  /* Start counting */
  if ( (retval = PAPI_start(EventSet)) != PAPI_OK )
    papi_error("PAPI_start", retval);
}

void papi_stop(const char *msg) {
  int retval, i;
  char EventCodeStr[PAPI_MAX_STR_LEN];
  int Events[nevents];
  long_long values[nevents];

  /* read values */
  if ( (retval = PAPI_stop(EventSet, values)) != PAPI_OK )
    papi_error("PAPI_stop", retval);

  /* List the events in the Event Set */
  i = nevents;            /* Can also use list_events to count them */
  if ( (retval = PAPI_list_events(EventSet, Events, &i)) != PAPI_OK )
    papi_error("PAPI_list_events", retval);
  if ( i != nevents )
    papi_error("Wrong number of events", 0);

  if ( msg ) printf("%s", msg);
  // Create padding
  int paddinglen = msg ? strlen(msg) : 0;
  char padding[paddinglen+1];
  for ( i = 0; i <= paddinglen; i++ )
    padding[i] = i == paddinglen ? 0 : ' ';

  for ( i = 0; i < nevents; i++ ) {
    if ( (retval = PAPI_event_code_to_name(Events[i], EventCodeStr))
         != PAPI_OK )
      papi_error("PAPI_event_code_to_name", retval);
    printf("%s%-15s %10lld\n", i > 0 ? padding : "", EventCodeStr,
           values[i]);
  }
}
#else
void papi_start() {}
void papi_stop(const char *msg) {}
#endif
