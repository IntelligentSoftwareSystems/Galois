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

#include "DynamicLib.h"
#include "Analysis.hpp"
#include "Node.hpp"
#include "Element.hpp"
#include "Mesh.hpp"

#include "EquationSystem.h"

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/CilkInit.h"
#include "galois/Timer.h"
#include "galois/runtime/TreeExec.h"

#include "Lonestar/BoilerPlate.h"

#include "llvm/Support/CommandLine.h"

#include <sys/time.h>

#include "CILK.hpp"
#include "GaloisDag.hpp"
#include "Seq.hpp"

#ifdef WITH_PAPI
#include "papi.h"
#endif

enum Schedulers { CILK, GALOIS_DAG, SEQ };

const char* const name = "DAGSolver";
const char* const desc = "Mesh-based FEM solver";
const char* const url  = NULL;

namespace cll = llvm::cl;

static cll::opt<std::string>
    prodlib("prodlib", cll::desc("Shared library with productions code"),
            cll::init("./pointproductions.so"));

static cll::opt<std::string>
    treefile("treefile", cll::desc("File with tree definition"), cll::init(""));

static cll::opt<std::string> matrixfile("matrixfile",
                                        cll::desc("File with frontal matrices"),
                                        cll::init(""));

static cll::opt<std::string>
    outtreefile("outtreefile", cll::desc("Output tree file"), cll::init(""));

static cll::opt<bool> debug("debug", cll::desc("Debug mode"), cll::init(false));

static cll::opt<Schedulers> scheduler("scheduler", cll::desc("Scheduler"),
                                      cll::values(
#ifdef HAVE_CILK
                                          clEnumVal(CILK, "Cilk-based"),
#endif
                                          clEnumVal(GALOIS_DAG, "Galois-DAG"),
                                          clEnumVal(SEQ, "Sequential"),
                                          clEnumValEnd),
                                      cll::init(CILK));

static cll::opt<SolverMode> solverMode(
    "solverMode", cll::desc("Elimination method"),
    cll::values(clEnumVal(OLD, "Old, hand-made elimination"),
                clEnumVal(LU, "LAPACK-based LU"),
                clEnumVal(CHOLESKY, "LAPACK-based Cholesky"), clEnumValEnd),
    cll::init(OLD));

#ifdef WITH_PAPI
static cll::opt<bool> papi_supported("perfcounters",
                                     cll::desc("Enable performance counters"),
                                     cll::init(false));
#endif

using namespace std;

void print_time(char* msg, timeval* t1, timeval* t2) {
  printf("%s: %f\n", msg,
         ((t2->tv_sec - t1->tv_sec) * 1000000 + (t2->tv_usec - t1->tv_usec)) /
             1000000.0);
}

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  struct timeval t1, t2;

#ifdef WITH_PAPI
  int events[5]       = {PAPI_FP_OPS, PAPI_LD_INS, PAPI_SR_INS, PAPI_L1_DCM,
                   PAPI_L2_TCM};
  long long values[5] = {
      0,
  };

  int eventSet = PAPI_NULL;

  int papi_err;
  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
    fprintf(stderr, "PAPI is unsupported.\n");
    papi_supported = false;
  }

  if (PAPI_num_counters() < 5) {
    fprintf(stderr, "PAPI is unsupported.\n");
    papi_supported = false;
  }

  if ((papi_err = PAPI_create_eventset(&eventSet)) != PAPI_OK) {
    fprintf(stderr, "Could not create event set: %s\n",
            PAPI_strerror(papi_err));
  }

  for (int i = 0; i < 5; ++i) {
    if ((papi_err = PAPI_add_event(eventSet, events[i])) != PAPI_OK) {
      fprintf(stderr, "Could not add event: %s\n", PAPI_strerror(papi_err));
    }
  }

#endif

  printf("Singularity solver - run info:\n");
  printf("\tmesh file: %s\n", treefile.c_str());

  Mesh* m = Mesh::loadFromFile(treefile.c_str());
  if (m == NULL) {
    printf("Could not load the mesh. Exiting.\n");
    exit(1);
  }

  printf("Analysis part.\n");
  gettimeofday(&t1, NULL);
  Analysis::enumerateDOF(m);
  gettimeofday(&t2, NULL);
  print_time("\tDOF enumeration", &t1, &t2);

  if (outtreefile.size()) {
    m->saveToFile(outtreefile.c_str());
    exit(0);
  }

  gettimeofday(&t1, NULL);
  Analysis::doAnalise(m);
  gettimeofday(&t2, NULL);
  print_time("\tanalysis", &t1, &t2);

  printf("\tnumber of elements: %lu\n", m->getElements().size());
  printf("\tproblem size (dofs): %lu\n", m->getDofs());
  if (debug) {
    Analysis::printTree(m->getRootNode());

    for (Element* e : m->getElements()) {
      Analysis::printElement(e);
    }
  }

  printf("Solving part.\n");

  printf("Root size: %d\n", m->getRootNode()->getDofs().size());

  // ALLOCATION
  gettimeofday(&t1, NULL);
  if (scheduler == GALOIS_DAG) {
    galoisAllocation(m->getRootNode(), solverMode);
  } else if (scheduler == CILK) {
#ifdef HAVE_CILK
    galois::CilkInit();
    cilk_alloc_tree(m->getRootNode(), solverMode);
#else
    printf("CILK is not supported.\n");
    return 1;
#endif
  } else if (scheduler == SEQ) {
    seqAllocation(m->getRootNode(), solverMode);
  }
  gettimeofday(&t2, NULL);
  print_time("\tallocation", &t1, &t2);

  // FACTORIZATION

#ifdef WITH_PAPI
  if (papi_supported) {
    if ((papi_err = PAPI_start(eventSet)) != PAPI_OK) {
      fprintf(stderr, "Could not start counters: %s\n",
              PAPI_strerror(papi_err));
    }
  }
#endif
  gettimeofday(&t1, NULL);
  if (scheduler == GALOIS_DAG) {
    galoisElimination(m->getRootNode());
  } else if (scheduler == CILK) {
#ifdef HAVE_CILK
    cilk_do_elimination(m->getRootNode());
#else
    printf("CILK is not supported.\n");
    return 1;
#endif
  } else if (scheduler == SEQ) {
    seqElimination(m->getRootNode());
  }
  gettimeofday(&t2, NULL);
  print_time("\tfactorization", &t1, &t2);

#ifdef WITH_PAPI
  if (papi_supported) {
    if ((papi_err = PAPI_stop(eventSet, values)) != PAPI_OK) {
      fprintf(stderr, "Could not get values: %s\n", PAPI_strerror(papi_err));
    }
    // PAPI_FP_OPS
    // PAPI_LD_INS
    // PAPI_SR_INS
    // PAPI_L1_DCM
    // PAPI_L2_TCM
    printf("Performance counters for factorization stage: \n");
    printf("\tFP OPS: %ld\n", values[0]);
    printf("\tLD INS: %ld\n", values[1]);
    printf("\tSR INS: %ld\n", values[2]);
    printf("\tL1 DCM: %ld\n", values[3]);
    printf("\tL2 TCM: %ld\n", values[4]);
  }
#endif

  // SOLUTION

  gettimeofday(&t1, NULL);
  if (scheduler == GALOIS_DAG) {
    galoisBackwardSubstitution(m->getRootNode());
  } else if (scheduler == CILK) {
#ifdef HAVE_CILK
    cilk_do_backward_substitution(m->getRootNode());
#else
    printf("CILK is not supported.\n");
#endif
  } else if (scheduler == SEQ) {
    seqBackwardSubstitution(m->getRootNode());
  }
  gettimeofday(&t2, NULL);
  print_time("\tsolution", &t1, &t2);

  delete m;
  // delete lib;

  return 0;
}
