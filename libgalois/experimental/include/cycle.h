#ifndef EXP_CYCLE_H
#define EXP_CYCLE_H

#if 1
typedef int ticks;
static double elapsed(ticks, ticks) { return 0; }
static ticks getticks() { return 0; }
static void printticks(const char* prefix, int count, ...) { }
#else
#include "cycle_main.h"
#include <stdio.h>
#include <cstdarg>
static void printticks(const char* prefix, int count, ...) {
  va_list ap;
  va_start(ap, count);
  printf("%s:", prefix);
  if (count > 0) {
    ticks prev = va_arg(ap, ticks);
    for (int i = 1; i < count; ++i) {
      ticks cur = va_arg(ap, ticks);
      printf(" %d: %.1f", i, elapsed(cur, prev));
      prev = cur;
    }
  }
  va_end(ap);
  printf("\n");
}

#endif

#endif
