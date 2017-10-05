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

static uint64_t numBytesRead = 0;
//static uint64_t* outIndex = nullptr;
//static uint32_t* edgeDest = nullptr;

uint64_t mpi_out_index(MPI_File mpiFile, uint64_t nodeIndex) {
  uint64_t position = (4 + nodeIndex) * sizeof(uint64_t);

  uint64_t retval;

  MPI_Status mpiStat;
  MPI_File_read_at(mpiFile, position, reinterpret_cast<char*>(&retval), 
                   sizeof(uint64_t), MPI_BYTE, &mpiStat); 
  numBytesRead += sizeof(uint64_t);
  return retval;
}

uint64_t mpi_edge_begin(MPI_File mpiFile, uint64_t nodeIndex) {
  if (nodeIndex != 0) {
    return mpi_out_index(mpiFile, nodeIndex - 1);
  } else {
    return 0;
  }
}

uint64_t mpi_edge_end(MPI_File mpiFile, uint64_t nodeIndex) {
  return mpi_out_index(mpiFile, nodeIndex);
}

uint64_t mpi_edge_dest(MPI_File mpiFile, uint64_t edgeIndex, uint64_t numNodes) {
  // ASSUMES V1 GRAPH WHERE SIZE OF EDGE IS UINT32
  uint64_t position = (4 + numNodes) * sizeof(uint64_t) + 
                      edgeIndex * sizeof(uint32_t);

  uint64_t retval;

  MPI_Status mpiStat;
  MPI_File_read_at(mpiFile, position, reinterpret_cast<char*>(&retval), 
                   sizeof(uint32_t), MPI_BYTE, &mpiStat); 
  numBytesRead += sizeof(uint32_t);
  return retval;
}


int main(int argc, char** argv) {
  galois::SharedMemSys G;

  MPI_Init(&argc, &argv);

  int hostID = 0;
  int numHosts = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &hostID);
  MPI_Comm_size(MPI_COMM_WORLD, &numHosts);

  if (hostID == 0) {
    printf("Graph to read: %s\n", argv[1]);
  }

  galois::graphs::OfflineGraph g(argv[1]);
  uint64_t numGlobalNodes = g.size();
  auto nodeSplit = galois::block_range((uint64_t)0, numGlobalNodes, hostID, 
                                       numHosts);
  printf("[%d] Get nodes %lu to %lu\n", hostID, nodeSplit.first, 
                                       nodeSplit.second);

  uint64_t nodeBegin = nodeSplit.first;
  uint64_t nodeEnd = nodeSplit.second;

  galois::DynamicBitSet ghosts;
  ghosts.resize(numGlobalNodes);

  MPI_File mpiFile;
  MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, 
                MPI_INFO_NULL, &mpiFile);

  galois::Timer timer;
  timer.start();

  // vector to hold a prefix sum for use in thread work distribution
  std::vector<uint64_t> prefixSumOfEdges(nodeEnd - nodeBegin);

  auto edgeOffset = mpi_edge_begin(mpiFile, nodeBegin);

  //uint64_t position = 4 * sizeof(uint64_t);
  //outIndex = (uint64_t*)malloc(sizeof(uint64_t)*numGlobalNodes);
  //edgeDest = (uint32_t*)malloc(sizeof(uint32_t) * g.sizeEdges());
  //assert(outIndex != nullptr);
  //assert(edgeDest != nullptr);

  //MPI_Status mpiStat;
  //MPI_File_read_at(mpiFile, position, (char*)outIndex, 
  //                 sizeof(uint64_t)*numGlobalNodes, MPI_BYTE, &mpiStat); 

  //position = (4 + numGlobalNodes) * sizeof(uint64_t);

  //MPI_File_read_at(mpiFile, position, (char*)edgeDest, 
  //                 sizeof(uint32_t)*g.sizeEdges(), MPI_BYTE, &mpiStat); 


  galois::do_all(galois::iterate(nodeBegin, nodeEnd),
    [&] (auto n) {
      //printf("asdf\n");
      auto ii = mpi_edge_begin(mpiFile, n);
      auto ee = mpi_edge_end(mpiFile, n);
      for (; ii < ee; ++ii) {
        ghosts.set(mpi_edge_dest(mpiFile, ii, numGlobalNodes));
      }
      prefixSumOfEdges[n - nodeBegin] = edgeOffset - ee;
    },
    galois::loopname("EdgeInspection"),
    galois::timeit(),
    galois::no_stats()
  );

  timer.stop();
  fprintf(stderr, "[%d] Edge inspection time : %f seconds to read %lu bytes (%f MBPS)\n", 
      hostID, timer.get_usec()/1000000.0f, numBytesRead, 
      numBytesRead/(float)timer.get_usec());

  return 0;
}
