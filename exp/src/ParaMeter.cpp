/** ParaMeter runtime -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Implementation of ParaMeter runtime
 * Ordered with speculation not supported yet
 *
 * @author Amber Hassaan <ahassaan@ices.utexas.edu>
 */
#include "Galois/Runtime/ParaMeter.h"

#include <cstdlib>
#include <ctime>
#include <cstdio>

namespace {
const size_t NAME_SIZE = 256;

struct Init {
  char name[NAME_SIZE];

  std::ostream& printHeader(std::ostream& out) {
    out << "LOOPNAME, STEP, AVAIL_PARALLELISM, WORKLIST_SIZE\n";
    return out;
  }

  static void genName(char* str, size_t size) {
    time_t rawtime;
    struct tm* timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(str, size, "ParaMeter_Stats_%Y-%m-%d_%H:%M:%S.csv", timeinfo);
  }

  Init() {
    genName(name, NAME_SIZE);
    std::ofstream statsFile(name, std::ios_base::out);
    printHeader(statsFile);
    statsFile.close();
  }
};

Init iii;
}

const char* GaloisRuntime::ParaMeter::statsFilename() {
  return iii.name;
}

