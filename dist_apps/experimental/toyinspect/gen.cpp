#include <cstdio>
#include <ctime>
#include <iostream>
#include <limits>
#include <iterator>
#include <mpi.h>
#include <galois/Timer.h>
#include "galois/graphs/OfflineGraph.h"
#include "galois/gstl.h"
#include "galois/runtime/Range.h"
#include "galois/Galois.h"
#include "galois/DistGalois.h"
#include "galois/graphs/MPIGraph.h"

static uint64_t numBytesRead = 0;
static uint64_t count        = 0;
static uint64_t* outIndex    = nullptr;
static uint32_t* edgeDest    = nullptr;

int main(int argc, char** argv) {
  galois::SharedMemSys G;

  MPI_Init(&argc, &argv);

  if (argc >= 3) {
    galois::setActiveThreads(std::stoi(argv[2]));
  }

  int hostID   = 0;
  int numHosts = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &hostID);
  MPI_Comm_size(MPI_COMM_WORLD, &numHosts);

  if (hostID == 0) {
    printf("Graph to read: %s\n", argv[1]);
  }

  galois::graphs::OfflineGraph g(argv[1]);
  uint64_t numGlobalNodes = g.size();
  auto nodeSplit =
      galois::block_range((uint64_t)0, numGlobalNodes, hostID, numHosts);
  printf("[%d] Get nodes %lu to %lu\n", hostID, nodeSplit.first,
         nodeSplit.second);

  uint64_t nodeBegin = nodeSplit.first;
  uint64_t nodeEnd   = nodeSplit.second;

  galois::DynamicBitSet ghosts;
  ghosts.resize(numGlobalNodes);

  std::vector<uint64_t> prefixSumOfEdges(nodeEnd - nodeBegin);

  galois::Timer timer;
  timer.start();

  galois::graphs::MPIGraph<void> test;
  test.loadPartialGraph(std::string(argv[1]), nodeBegin, nodeEnd,
                        *g.edge_begin(nodeBegin), *g.edge_begin(nodeEnd),
                        numGlobalNodes);

  auto edgeOffset = test.edgeBegin(nodeBegin);

  galois::do_all(galois::iterate(nodeBegin, nodeEnd),
                 [&](auto n) {
                   auto ii = test.edgeBegin(n);
                   auto ee = test.edgeEnd(n);

                   for (; ii < ee; ++ii) {
                     ghosts.set(test.edgeDestination(*ii));
                   }
                   prefixSumOfEdges[n - nodeBegin] = edgeOffset - ee;
                 },
                 galois::no_stats(), galois::loopname("EdgeInspection"));

  timer.stop();

  fprintf(
      stderr,
      "[%d] Edge inspection time : %f seconds to read %lu bytes (%f MBPS)\n",
      hostID, timer.get_usec() / 1000000.0f, test.getBytesRead(),
      test.getBytesRead() / (float)timer.get_usec());

  printf("count is %lu\n", count);
  free(outIndex);
  free(edgeDest);

  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}
