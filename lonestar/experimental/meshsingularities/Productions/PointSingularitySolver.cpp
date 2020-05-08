/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include <stdio.h>
#include <iostream>

#include "Production.h"
#include "GaloisWorker.h"

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/UnionFind.h"
#include "galois/Graph/LCGraph.h"
#include "galois/ParallelSTL/ParallelSTL.h"
#include "galois/Graph/LC_Morph_Graph.h"

#include "Lonestar/BoilerPlate.h"

#include "llvm/Support/CommandLine.h"

#include "Functions.h"
#include "TaskDescription.h"

const char* const name = "Mesh singularities";
const char* const desc = "Compute the solution of differential equation";
const char* const url  = NULL;

namespace cll = llvm::cl;

static cll::opt<int> tiers("tiers", cll::desc("Number of Tiers"), cll::init(2));
static cll::opt<int> dimensions("dimensions", cll::desc("Dimensions"),
                                cll::init(2));
static cll::opt<int> polynomial("polynomial", cll::desc("Polynomial degree"),
                                cll::init(2));

static cll::opt<Functions>
    function("function", cll::desc("Choose a function:"),
             cll::values(clEnumVal(F1, "f1(x, [y, [z]]) = 1.0"),
                         clEnumVal(F2, "f2(x, [y, [z]]) = x+[y+[z]]"),
                         clEnumVal(F3, "f3(x, [y, [z]]) = x^2+[y^2+[z^2]]"),
                         clEnumVal(F4, "f4(x, [y, [z]]) = x^3+[y^3+[z^3]]"),
                         clEnumValEnd),
             cll::init(F1));

static cll::opt<Singularities> singularity(
    "singularity", cll::desc("Singularity type:"),
    cll::values(clEnumVal(POINT, "Point singularity"),
                clEnumVal(CENTRAL_POINT, "Cetral-point singularity"),
                clEnumVal(EDGE, "Edge singularity"),
                clEnumVal(FACE, "Face singularity"),
                clEnumVal(ANISOTROPIC, "Anisotropic mesh"), clEnumValEnd),
    cll::init(EDGE));

static cll::opt<double>
    coord_x("coord_x",
            cll::desc("X coordinate for left-bottom vertex of shape"),
            cll::init(0.0));
static cll::opt<double>
    coord_y("coord_y",
            cll::desc("Y coordinate for left-bottom vertex of shape"),
            cll::init(0.0));
static cll::opt<double>
    coord_z("coord_z",
            cll::desc("Z coordinate for left-bottom vertex of shape"),
            cll::init(0.0));

static cll::opt<double> size("size", cll::desc("Size of the shape"),
                             cll::init(1.0));
static cll::opt<bool>
    performTests("performTests",
                 cll::desc("Run extended tests of computed solution"),
                 cll::init(true));

static cll::opt<bool> quad("quad",
                           cll::desc("Special case for edge singularity"),
                           cll::init(false));
static cll::opt<std::string>
    productions("productions",
                cll::desc("Shared library with productions code:"),
                cll::init("pointproductions.so"));

static cll::opt<Schedulers>
    scheduler("scheduler",
              cll::desc("Scheduler used for solver to parallelize execution"),
              cll::values(clEnumVal(OLD, "Old queue-based scheduler"),
                          clEnumVal(CILK, "CILK version"),
                          clEnumVal(GALOIS_DAG, "Galois-DAG scheduler")),
              cll::init(OLD));

//#define WITH_MUMPS_ENABLED 0

#ifdef WITH_MUMPS_ENABLED
static cll::opt<bool> mumps("mumps", cll::desc("Pass data to MUMPS"),
                            cll::init(false));
#endif

template <typename Algorithm>
std::vector<double>* run(TaskDescription& td) {
  galois::StatTimer U(name);
  Algorithm algorithm;
  U.start();
  std::vector<double>* res = algorithm(td);
  U.stop();
  return res;
}

#ifdef WITH_MUMPS_ENABLED
extern int execute_mumps(int argc, char** argv,
                         TaskDescription& taskDescription);
#endif

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);

  TaskDescription taskDescription;

  taskDescription.dimensions       = dimensions;
  taskDescription.polynomialDegree = polynomial;
  taskDescription.function         = functionsTable[function].func;
  taskDescription.nrOfTiers        = tiers;
  taskDescription.size             = size;
  taskDescription.x                = coord_x;
  taskDescription.y                = coord_y;
  taskDescription.z                = coord_z;
  taskDescription.quad             = quad;
  taskDescription.singularity      = singularity;
  taskDescription.scheduler        = scheduler;
  taskDescription.performTests     = performTests;

#ifdef WITH_MUMPS_ENABLED
  if (mumps)
    execute_mumps(argc, argv, taskDescription);
  else
#endif
  {
    std::vector<double>* result = run<ProductionProcess>(taskDescription);
    // in result we have an output from solver,
    // do whatever you want with it.
    delete result;
  }
  return 0;
}
