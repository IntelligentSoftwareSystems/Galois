/*
   failfast.h

   Implements debug routines. Part of the GGC source code.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu>
*/

#pragma once

#include <stdio.h>
#include <stdarg.h>

static void ff_fprintf(const char* file, const int line, FILE* stream,
                       const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int err = vfprintf(stream, fmt, ap);
  if (err < 0) {
    fprintf(stderr, "%s:%d:fprintf failed.\n", file, line);
    exit(1);
  }
}

#define check_fprintf(...) ff_fprintf(__FILE__, __LINE__, __VA_ARGS__)
