#ifndef _GALOIS_CUSP_
#define _GALOIS_CUSP_

#include "galois/DistGalois.h"
#include "galois/graphs/DistributedGraph.h"
#include "galois/graphs/NewGeneric.h"
#include "galois/graphs/GenericPartitioners.h"

namespace galois {
  //! TODO doxygen
  enum CUSP_GRAPH_TYPE {
    CUSP_CSR,
    CUSP_CSC
  };
  
  /**
   * TODO doxygen
   *
   * TODO @todo look into making void node data work in LargeArray for D-Galois;
   * void specialization. For now, use char as default
   */
  template<typename PartitionPolicy, typename NodeData=char,
           typename EdgeData=void>
  galois::graphs::DistGraph<NodeData, EdgeData>* cuspPartitionGraph(
        std::string graphFile, CUSP_GRAPH_TYPE inputType,
        CUSP_GRAPH_TYPE outputType, bool symmetricGraph=false,
        std::string transposeGraphFile="",
        bool cuspAsync=true, uint32_t cuspStateRounds=100,
        galois::graphs::MASTERS_DISTRIBUTION readPolicy=galois::graphs::BALANCED_EDGES_OF_MASTERS,
        uint32_t nodeWeight=0, uint32_t edgeWeight=0
  ) {
    auto& net = galois::runtime::getSystemNetworkInterface();
    using DistGraphConstructor = galois::graphs::NewDistGraphGeneric<NodeData,
                                    EdgeData, PartitionPolicy>;

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

      return new DistGraphConstructor(inputToUse, net.ID, net.Num, cuspAsync,
                                      cuspStateRounds, useTranspose, readPolicy,
                                      nodeWeight, edgeWeight);
    } else {
      // symmetric graph path: assume the passed in graphFile is a symmetric
      // graph; output is also symmetric
      return new DistGraphConstructor(graphFile, net.ID, net.Num, cuspAsync,
                                      cuspStateRounds, false, readPolicy,
                                      nodeWeight, edgeWeight);
    }
  }
} // end namespace galois
#endif
