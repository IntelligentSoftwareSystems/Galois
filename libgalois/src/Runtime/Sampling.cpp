/** Sampling implementation -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/Sampling.h"
//#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/EnvCheck.h"
#include "Galois/Runtime/ThreadPool.h"
//#include "Galois/Substrate/gio.h"
#include <cstdlib>

static bool isOn = false;

static void endPeriod() {
  assert(isOn);
  isOn = false;
  int val;
  static bool doexit = Galois::Runtime::EnvCheck("GALOIS_EXIT_AFTER_SAMPLING", val);
  if (doexit)
    exit(val);
}

static void beginPeriod() {
  assert(!isOn);
  isOn = true;
  int val;
  static bool doexit = Galois::Runtime::EnvCheck("GALOIS_EXIT_BEFORE_SAMPLING", val);
  if (doexit) {
    exit(val);
  }
}

#ifdef GALOIS_USE_VTUNE
#include "ittnotify.h"
#include <map>

namespace vtune {

static std::map<std::string, __itt_domain*> domains;
static __itt_domain* cur_pD = nullptr;

static void begin(const char* loopname) {
  static bool dovtune = Galois::Runtime::EnvCheck("GALOIS_SAMPLE_VTUNE");
  if (!dovtune)
    return;
  auto& pD = domains[loopname];
  if (!pD) {
    pD = __itt_domain_create(loopname);
    pD->flags = 1; // enable domain
  }
  cur_pD = pD;
  __itt_frame_begin_v3(pD, NULL);
}

static void end() {
  static bool dovtune = Galois::Runtime::EnvCheck("GALOIS_SAMPLE_VTUNE");
  if (!dovtune)
    return;
  __itt_frame_end_v3(cur_pD, NULL);
}
}
#else
namespace vtune {
static void begin(const char*) {}
static void end() {}
}
#endif

#ifdef GALOIS_USE_HPCTOOLKIT
#include <hpctoolkit.h>
#include "Galois/Runtime/ll/TID.h"

namespace hpctoolkit {
static void begin() {
  static bool dohpc = Galois::Runtime::EnvCheck("GALOIS_SAMPLE_HPC");
  if (!dohpc)
    return;
  hpctoolkit_sampling_start();
}

static void end() {
  static bool dohpc = Galois::Runtime::EnvCheck("GALOIS_SAMPLE_HPC");
  if (!dohpc)
    return;
  hpctoolkit_sampling_stop();
}
}
#else
namespace hpctoolkit {
static void begin() {}
static void end() {}
}
#endif

void Galois::Runtime::beginSampling(const char* loopname) {
  assert(Galois::Runtime::ThreadPool::getTID() == 0);
  beginPeriod();
  vtune::begin(loopname);
  hpctoolkit::begin();
}

void Galois::Runtime::endSampling() {
  assert(Galois::Runtime::ThreadPool::getTID() == 0);
  hpctoolkit::end();
  vtune::end();
  endPeriod();
}
