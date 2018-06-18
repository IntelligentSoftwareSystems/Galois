/* Time measurement (from HW1, HW3) */
#include <sys/time.h>
#define TIMEDELTA(a, b, small, multiplier)                                     \
  (((b).tv_sec - (a).tv_sec) * multiplier + ((b).small - (a).small))
#define SEC_IN_MICROSEC 1000000
#define TIMEDELTA_MICRO(a, b) TIMEDELTA(a, b, tv_usec, SEC_IN_MICROSEC)

#include "dag_graphs.h"
#ifndef SERIAL
#include "dag_lw.h"
#include "dag_algo.h"
#endif
