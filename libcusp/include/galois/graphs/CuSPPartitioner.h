#ifndef _GALOIS_CUSP_
#define _GALOIS_CUSP_

#include "galois/graphs/DistributedGraph.h"
#include "galois/graphs/NewGeneric.h"
#include "galois/graphs/Generic.h"

namespace galois {
  //! TODO doxygen
  enum CUSP_GRAPH_TYPE {
    CUSP_CSR,
    CUSP_CSC
  }
  
  /**
   * TODO doxygen
   *
   *
   * TODO @todo look into making void node data work in LargeArray for D-Galois;
   * void specialization. For now, use char as default
   */
  template<typename PartitionPolicy, typename NodeData=char, typename EdgeData=void>
  DistGraph<NodeData, EdgeData>* cuspPartitionGraph(std::string graphFile,
        GRAPH_TYPE inputType, GRAPH_TYPE outputType, bool symmetricGraph=false,
        std::string transposeGraphFile="") {
    auto& net = galois::runtime::getSystemNetworkInterface();
    using DistGraphConstructor = NewDistGraphGeneric<NodeData, EdgeData,
                                                     PartitionPolicy>;
    // TODO @todo bring back graph saving/reading functionality?
    std::string localGraphName = "";

    if (!symmetricGraph) {
      // out edges or in edges
      std::string inputToUse;
      // depending on output type may need to transpose edges
      bool useTranspose;

      // see what input is specified
      if (inputType == CUSP_CSR) {
        inputToUse = graphFile;
        if (outputType == CUSP_CSR) {
          useTranspose = false;
        } else if (outputType == CUSP_CSC) {
          useTranspose = true;
        } else {
          GALOIS_DIE("CuSP output graph type is invalid");
        }
      } else if (inputType == CUSP_CSC) {
        inputToUse = transposeGraphFile;
        if (outputType == CUSP_CSR) {
          useTranspose = true;
        } else if (outputType == CUSP_CSC) {
          useTranspose = false;
        } else {
          GALOIS_DIE("CuSP output graph type is invalid");
        }
      } else {
        GALOIS_DIE("Invalid input graph type specified in CuSP partitioner");
      }

      return new DistGraphConstructor(inputToUse, net.ID, net.Num, useTranspose,
                                      false, localGraphName);
    } else {
      // symmetric graph path: assume the passed in graphFile is a symmetric
      // graph; output is also symmetric
      return new DistGraphConstructor(inputFile, net.ID, net.Num, false, false,
                                      localGraphName);
    }
  }
} // end namespace galois
