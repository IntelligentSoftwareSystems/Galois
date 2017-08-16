/** dGraph loader -*- C++ -*-
 * @file
 * dGraphLoader.h
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 * Command line arguments and functions for loading dGraphs into memory
 *
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#ifndef D_GRAPH_LOADER
#define D_GRAPH_LOADER

#include "Lonestar/BoilerPlate.h"
#include "Galois/Runtime/dGraph_edgeCut.h"
#include "Galois/Runtime/dGraph_cartesianCut.h"
#include "Galois/Runtime/dGraph_hybridCut.h"

/*******************************************************************************
 * Supported partitioning schemes
 ******************************************************************************/
enum PARTITIONING_SCHEME {
  OEC, IEC, PL_VCUT, CART_VCUT
};

/*******************************************************************************
 * Graph-loading-related command line arguments
 ******************************************************************************/
namespace cll = llvm::cl;

static cll::opt<std::string> inputFile(cll::Positional, 
                                       cll::desc("<input file>"), 
                                       cll::Required);
static cll::opt<std::string> inputFileTranspose("graphTranspose",
                                       cll::desc("<input file, transposed>"), 
                                       cll::init(""));
static cll::opt<bool> inputFileSymmetric("symmetricGraph",
                                       cll::desc("Set this flag if graph is symmetric"), 
                                       cll::init(false));
static cll::opt<std::string> partFolder("partFolder", 
                                        cll::desc("path to partitionFolder"), 
                                        cll::init(""));
static cll::opt<PARTITIONING_SCHEME> partitionScheme("partition",
                                     cll::desc("Type of partitioning."),
                                     cll::values(
                                       clEnumValN(OEC, "oec", 
                                                  "Outgoing edge cut"), 
                                       clEnumValN(IEC, "iec", 
                                                  "Incoming edge cut"), 
                                       clEnumValN(PL_VCUT, "pl_vcut", 
                                                  "Powerlyra Vertex Cut"), 
                                       clEnumValN(CART_VCUT , "cart_vcut", 
                                                  "Cartesian Vertex Cut"), 
                                       clEnumValEnd),
                                     cll::init(OEC));
static cll::opt<unsigned int> VCutThreshold("VCutThreshold", 
                                            cll::desc("Threshold for high "
                                                      "degree edges."), 
                                            cll::init(1000));

/*******************************************************************************
 * Graph-loading functions
 ******************************************************************************/
/**
 * Loads a symmetric graph file (i.e. directed graph with edges in both 
 * directions)
 *
 * @tparam NodeData node data to store in graph
 * @tparam EdgeData edge data to store in graph
 * @param scaleFactor How to split nodes among hosts
 * @returns a pointer to a newly allocated hGraph based on the command line
 * loaded based on command line arguments
 */
template<typename NodeData, typename EdgeData>
hGraph<NodeData, EdgeData>* constructSymmetricGraph(std::vector<unsigned> 
                                                    scaleFactor) {
  if (!inputFileSymmetric) {
    GALOIS_DIE("Calling constructSymmetricGraph without inputFileSymmetric "
               "flag");
  }

  typedef hGraph_edgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef hGraph_vertexCut<NodeData, EdgeData> Graph_vertexCut;
  typedef hGraph_cartesianCut<NodeData, EdgeData> Graph_cartesianCut;

  auto& net = Galois::Runtime::getSystemNetworkInterface();
  
  switch(partitionScheme) {
    case OEC:
    case IEC:
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, false);
    case PL_VCUT:
      return new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
                                 scaleFactor, false, VCutThreshold);
    case CART_VCUT:
      return new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false);
    default:
      GALOIS_DIE("Error: partition scheme specified is invalid");
      return nullptr;
  }
}

/**
 * Loads a graph file with the purpose of iterating over the out edges
 * of the graph.
 *
 * @tparam NodeData node data to store in graph
 * @tparam EdgeData edge data to store in graph
 * @tparam iterateOut says if you want to iterate over out edges or not; if
 * false, will iterate over in edgse
 * @tparam enable_if this function  will only be enabled if iterateOut is true
 * @param scaleFactor How to split nodes among hosts
 * @returns a pointer to a newly allocated hGraph based on the command line
 * loaded based on command line arguments
 */
template<typename NodeData, typename EdgeData, bool iterateOut = true,
         typename std::enable_if<iterateOut>::type* = nullptr>
hGraph<NodeData, EdgeData>* constructGraph(std::vector<unsigned> scaleFactor) {
  typedef hGraph_edgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef hGraph_vertexCut<NodeData, EdgeData> Graph_vertexCut;
  typedef hGraph_cartesianCut<NodeData, EdgeData> Graph_cartesianCut;

  auto& net = Galois::Runtime::getSystemNetworkInterface();
  
  switch(partitionScheme) {
    case OEC:
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, false);
    case IEC:
      if (inputFileTranspose.size()) {
        return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                                 scaleFactor, true);
      } else {
        GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                   "graph");
        break;
      }
    case PL_VCUT:
      return new Graph_vertexCut(inputFile, partFolder, net.ID, net.Num, 
                                 scaleFactor, false, VCutThreshold);
    case CART_VCUT:
      return new Graph_cartesianCut(inputFile, partFolder, net.ID, net.Num, 
                                    scaleFactor, false);
    default:
      GALOIS_DIE("Error: partition scheme specified is invalid");
      return nullptr;
  }
}

/**
 * Loads a graph file with the purpose of iterating over the in edges
 * of the graph.
 *
 * @tparam NodeData node data to store in graph
 * @tparam EdgeData edge data to store in graph
 * @tparam iterateOut says if you want to iterate over out edges or not; if
 * false, will iterate over in edgse
 * @tparam enable_if this function  will only be enabled if iterateOut is false
 * (i.e. iterate over in-edges)
 * @param scaleFactor How to split nodes among hosts
 * @returns a pointer to a newly allocated hGraph based on the command line
 * loaded based on command line arguments
 */
template<typename NodeData, typename EdgeData, bool iterateOut = true,
         typename std::enable_if<!iterateOut>::type* = nullptr>
hGraph<NodeData, EdgeData>* constructGraph(std::vector<unsigned> scaleFactor) {
  typedef hGraph_edgeCut<NodeData, EdgeData> Graph_edgeCut;
  typedef hGraph_vertexCut<NodeData, EdgeData> Graph_vertexCut;
  typedef hGraph_cartesianCut<NodeData, EdgeData> Graph_cartesianCut;

  auto& net = Galois::Runtime::getSystemNetworkInterface();

  switch(partitionScheme) {
    case OEC:
      return new Graph_edgeCut(inputFile, partFolder, net.ID, net.Num, 
                               scaleFactor, true);
    case IEC:
      if (inputFileTranspose.size()) {
        return new Graph_edgeCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                                 scaleFactor, false);
      } else {
        GALOIS_DIE("Error: attempting incoming edge cut without transpose "
                   "graph");
        break;
      }
    case PL_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_vertexCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                                   scaleFactor, false, VCutThreshold);
      } else {
        GALOIS_DIE("Error: (plc) iterate over in-edges without transpose graph");
        break;
      }

    case CART_VCUT:
      if (inputFileTranspose.size()) {
        return new Graph_cartesianCut(inputFileTranspose, partFolder, net.ID, net.Num, 
                                      scaleFactor, false);
      } else {
        GALOIS_DIE("Error: (cvc) iterate over in-edges without transpose graph");
        break;
      }
    default:
      GALOIS_DIE("Error: partition scheme specified is invalid");
      return nullptr;
  }
}
#endif
