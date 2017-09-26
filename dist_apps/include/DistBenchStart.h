/** Common command line processing for benchmarks -*- C++ -*-
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
 * Copyright (C) 2017, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * Common benchmark initialization
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Loc Hoang <l_hoang@utexas.edu>
 */
#ifndef DIST_BENCH_START_H
#define DIST_BENCH_START_H

#include "galois/Galois.h"
#include "galois/Version.h"
#include "llvm/Support/CommandLine.h"
#include "galois/runtime/dGraphLoader.h"

#ifdef __GALOIS_HET_CUDA__
#include "galois/runtime/Cuda/cuda_device.h"
#include "galois/runtime/Cuda/cuda_context_decl.h"
#endif

//! standard global options to the benchmarks
namespace cll = llvm::cl;

extern cll::opt<bool> skipVerify;
extern cll::opt<int> numThreads;
extern cll::opt<int> numRuns;
extern cll::opt<bool> savegraph;
extern cll::opt<std::string> outputFile;
extern cll::opt<bool> verifyMax;
extern cll::opt<std::string> statFile;
extern cll::opt<unsigned int> enforce_metadata;
extern cll::opt<bool> verify;

#ifdef __GALOIS_HET_CUDA__
enum Personality {
   CPU, GPU_CUDA, GPU_OPENCL
};

std::string personality_str(Personality p);

extern cll::opt<int> gpudevice;
extern cll::opt<Personality> personality;
extern cll::opt<unsigned> scalegpu;
extern cll::opt<unsigned> scalecpu;
extern cll::opt<int> num_nodes;
extern cll::opt<std::string> personality_set;
#endif

void DistBenchStart(int argc, char** argv, const char* app, 
                    const char* desc = nullptr, const char* url = nullptr);

#ifdef __GALOIS_HET_CUDA__
void SetupHetero(std::vector<unsigned>& scaleFactor);
#endif



template <typename NodeData, typename EdgeData, bool iterateOutEdges = true>
hGraph<NodeData, EdgeData>* LoadDGraph(std::vector<unsigned>& scaleFactor,
                                       struct CUDA_Context** cuda_ctx) {
  galois::StatTimer dGraphTimer("TIMER_HG_INIT"); 
  dGraphTimer.start();

  hGraph<NodeData, EdgeData>* loadedGraph = 
      constructGraph<NodeData, EdgeData, iterateOutEdges>(scaleFactor);

  // graph marshalling
  #ifdef __GALOIS_HET_CUDA__
  auto& net = galois::runtime::getSystemNetworkInterface();
  const unsigned my_host_id = galois::runtime::getHostID();

  galois::StatTimer marshalTimer("TIMER_GRAPH_MARSHAL"); 
  marshalTimer.start();

  if (personality == GPU_CUDA) {
    *cuda_ctx = get_CUDA_context(my_host_id);

    if (!init_CUDA_context(*cuda_ctx, gpudevice)) {
      GALOIS_DIE("Failed to initialize CUDA context");
    }

    MarshalGraph m = (*loadedGraph).getMarshalGraph(my_host_id);
    load_graph_CUDA(*cuda_ctx, m, net.Num);
  } else if (personality == GPU_OPENCL) {
    //galois::opencl::cl_env.init(cldevice.Value);
  }

  marshalTimer.stop();
  #endif

  dGraphTimer.stop();

  return loadedGraph;
}

#endif
