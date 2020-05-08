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

#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include <set>
#include <vector>
#include <string>

#include "galois/graphs/FileGraph.h"
#include "galois/graphs/OfflineGraph.h"

static const char* const name = "Off-line graph partitioner";
static const char* const desc =
    "A collection of routines to partition graphs off-line.";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int>
    numPartitions("num", cll::desc("Number of partitions to be created"),
                  cll::init(2));
static cll::opt<std::string> outputFolder(
    "outputTo",
    cll::desc("Name of the output folder to store the partitioned graphs."),
    cll::init("./"));
static cll::opt<std::string> prefixTmp(
    "prefixTmp",
    cll::desc("prefix for tmp dir where intermediate files will be stored."),
    cll::Required);

typedef galois::graphs::OfflineGraph GraphType;
typedef GraphType::edge_iterator EdgeItType;
typedef GraphType::iterator NodeItType;

#define _HAS_EDGE_DATA 1

#ifdef _HAS_EDGE_DATA
typedef unsigned int EdgeDataType;
#endif

#include "Common.h"
//#include "RandomPartitioner.h"
//#include "GreedyBalancedPartitioner.h"
#include "GreedyBalancedPartitionerDisk.h"
#include "GBalancedPartitionerDisk2.h"
// using namespace std;

int main(int argc, char** argv) {
  LonestarStart(argc, argv, name, desc, url);
  galois::Timer T_total, T_offlineGraph_init, T_DistGraph_init, T_init, T_HSSSP;
  T_total.start();
  T_DistGraph_init.start();
  galois::graphs::OfflineGraph g(inputFile);
  T_DistGraph_init.stop();
  // VertexCutInfo vci;
  T_init.start();
  {
    GBPD2 p;
    std::cout << "Temp files will be stored in " << prefixTmp << "\n";
    p(outputFolder, g, numPartitions, prefixTmp);
  }
  T_init.stop();
  if (!verifyParitions(outputFolder, g, numPartitions)) {
    std::cout << "Verification of partitions failed! Contact developers!\n";
  } else {
    std::cout << "Partitions verified!\n";
  }
  std::cout << "Completed partitioning.\n";
  return 0;
}
