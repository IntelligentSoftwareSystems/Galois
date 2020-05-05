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

/*

 @Vinicius Possani
 Blif format writer, October 17, 2018.

*/

#include "BlifWriter.h"
#include "../util/utilString.h"
#include <unordered_set>

BlifWriter::BlifWriter() {}

BlifWriter::BlifWriter(std::string path) { setFile(path); }

BlifWriter::~BlifWriter() { blifFile.close(); }

void BlifWriter::setFile(std::string path) {
  this->path = path;
  blifFile.close();
  blifFile.open(path.c_str(), std::ios::trunc);
}

bool BlifWriter::isOpen() { return blifFile.is_open(); }

void BlifWriter::close() { blifFile.close(); }

void BlifWriter::writeNetlist(aig::Aig& aig, algorithm::PriCutManager& cutMan) {

  aig::GNode leaf;
  aig::Graph& aigGraph   = aig.getGraph();
  std::string designName = aig.getDesignName();
  find_and_replace(designName, " ", "_");
  int nDigitsPIs = countDigits(aig.getInputNodes().size() - 1);
  int nDigitsPOs = countDigits(aig.getOutputNodes().size() - 1);

  this->blifFile << ".model " << designName << std::endl;

  this->blifFile << ".inputs ";
  for (size_t i = 0; i < aig.getInputNodes().size(); i++) {
    this->blifFile << "pi" << std::setfill('0') << std::setw(nDigitsPIs) << i
                   << " ";
  }
  this->blifFile << std::endl;

  this->blifFile << ".outputs ";
  for (size_t i = 0; i < aig.getOutputNodes().size(); i++) {
    this->blifFile << "po" << std::setfill('0') << std::setw(nDigitsPOs) << i
                   << " ";
  }
  this->blifFile << std::endl;

  for (auto entry : cutMan.getCovering()) {
    this->blifFile << ".names";

    if (Functional32::isConstZero(cutMan.readTruth(entry.second),
                                  cutMan.getK())) {
      // Output
      this->blifFile << " n" << entry.first << std::endl << "0" << std::endl;
      continue;
    }

    if (Functional32::isConstOne(cutMan.readTruth(entry.second),
                                 cutMan.getK())) {
      // Output
      this->blifFile << " n" << entry.first << std::endl << "1" << std::endl;
      continue;
    }

    // Inputs
    for (int i = 0; i < entry.second->nLeaves; i++) {

      leaf                    = aig.getNodes()[entry.second->leaves[i]];
      aig::NodeData& leafData = aigGraph.getData(leaf);

      if (leafData.type == aig::NodeType::PI) {
        this->blifFile << " pi" << std::setfill('0') << std::setw(nDigitsPIs)
                       << (leafData.id - 1);
      } else {
        this->blifFile << " n" << leafData.id;
      }
    }
    // Output
    this->blifFile << " n" << entry.first << std::endl;
    // Cubes
    this->blifFile << Functional32::toCubeString(cutMan.readTruth(entry.second),
                                                 cutMan.getNWords(),
                                                 entry.second->nLeaves);
  }

  // Define PO poloarities
  for (size_t i = 0; i < aig.getOutputNodes().size(); i++) {

    auto inEdgeIt = aigGraph.in_edge_begin(aig.getOutputNodes()[i]);
    bool outEdgePolarity =
        aigGraph.getEdgeData(inEdgeIt, galois::MethodFlag::READ);
    aig::GNode inNode         = aigGraph.getEdgeDst(inEdgeIt);
    aig::NodeData& inNodeData = aigGraph.getData(inNode);

    if (inNodeData.type == aig::NodeType::PI) {
      this->blifFile << ".names pi" << std::setfill('0')
                     << std::setw(nDigitsPIs) << (inNodeData.id - 1);
      this->blifFile << " po" << std::setfill('0') << std::setw(nDigitsPOs) << i
                     << std::endl;
      this->blifFile << ((outEdgePolarity == true) ? "1 1" : "0 1")
                     << std::endl;
    } else {
      if (inNodeData.type == aig::NodeType::CONSTZERO) {
        this->blifFile << ".names "
                       << " po" << std::setfill('0');
        this->blifFile << std::setw(nDigitsPOs) << i << std::endl;
        this->blifFile << ((outEdgePolarity == true) ? "0" : "1") << std::endl;
      } else {
        this->blifFile << ".names n" << inNodeData.id;
        this->blifFile << " po" << std::setfill('0') << std::setw(nDigitsPOs)
                       << i << std::endl;
        this->blifFile << ((outEdgePolarity == true) ? "1 1" : "0 1")
                       << std::endl;
      }
    }
  }

  this->blifFile << ".end" << std::endl;
}

int BlifWriter::countDigits(int n) {

  int nDigits = 0;
  while (n) {
    n = n / 10;
    nDigits++;
  }
  return nDigits;
}
