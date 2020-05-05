/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#include "AigWriter.h"
#include "../util/utilString.h"

AigWriter::AigWriter() {}

AigWriter::AigWriter(std::string path) {
  setFile(path);
}

AigWriter::~AigWriter() { aigerFile.close(); }

void AigWriter::setFile(std::string path) {
  this->path = path;
  aigerFile.close();
  aigerFile.open(path.c_str(), std::ios::trunc);
}

bool AigWriter::isOpen() { return aigerFile.is_open(); }

void AigWriter::writeAag(Aig& aig) {
  aig.resetAndIds();
  writeAagHeader(aig);
  writeInputs(aig);
  writeLatchesAag(aig);
  writeOutputs(aig);
  writeAndsAag(aig);
  writeSymbolTable(aig);
}

void AigWriter::writeAagHeader(Aig& aig) {
  int i = aig.getNumInputs();
  int l = aig.getNumLatches();
  int o = aig.getNumOutputs();
  int a = aig.getNumAnds();
  int m = i + l + a;
  aigerFile << "aag " << m << " " << i << " " << l << " " << o << " " << a
            << std::endl;
}

void AigWriter::writeInputs(Aig& aig) {

  aig::Graph& graph = aig.getGraph();

  for (auto input : aig.getInputNodes()) {
    aig::NodeData& inputData = graph.getData(input, galois::MethodFlag::READ);
    aigerFile << inputData.id * 2 << std::endl;
  }
}

void AigWriter::writeLatchesAag(Aig& aig) {

  aig::Graph& aigGraph = aig.getGraph();

  for (aig::GNode latchNode : aig.getLatchNodes()) {
    aig::NodeData& latchNodeData =
        aigGraph.getData(latchNode, galois::MethodFlag::READ);
    aigerFile << latchNodeData.id * 2 << " ";
    auto inEdge         = aigGraph.in_edge_begin(latchNode);
    bool inEdgePolarity = aigGraph.getEdgeData(inEdge);
    aig::GNode inNode   = aigGraph.getEdgeDst(inEdge);
    aig::NodeData& inNodeData =
        aigGraph.getData(inNode, galois::MethodFlag::READ);
    // bool initState = latchNode->getInitialValue(); // FIXME;
    if (inEdgePolarity) {
      aigerFile << inNodeData.id * 2 << std::endl;
      // aigerFile << inNodeData.id << " " << initState << std::endl;
    } else {
      aigerFile << (inNodeData.id * 2) + 1 << std::endl;
      // aigerFile << inNodeData.id + 1 << " " << initState << std::endl;
    }
  }
}

void AigWriter::writeOutputs(Aig& aig) {

  aig::Graph& graph = aig.getGraph();

  for (auto output : aig.getOutputNodes()) {

    auto inEdge         = graph.in_edge_begin(output);
    bool inEdgePolarity = graph.getEdgeData(inEdge, galois::MethodFlag::READ);
    aig::GNode inNode   = graph.getEdgeDst(inEdge);
    aig::NodeData& inNodeData = graph.getData(inNode, galois::MethodFlag::READ);
    if (inEdgePolarity) {
      aigerFile << inNodeData.id * 2 << std::endl;
    } else {
      aigerFile << (inNodeData.id * 2) + 1 << std::endl;
    }
  }
}

void AigWriter::writeAndsAag(Aig& aig) {

  std::stack<aig::GNode> stack;

  aig.computeTopologicalSortForAnds(stack);

	//std::cout << "size: " << aig.getNumAnds() << std::endl;
	//std::cout << "stack size: " << stack.size() << std::endl;

  aig::Graph& graph = aig.getGraph();

  // unsigned int currentID = aig.getNumInputs() + aig.getNumLatches() + 1;

  while (!stack.empty()) {

    aig::GNode node = stack.top();
    stack.pop();

    aig::NodeData& nodeData = graph.getData(node, galois::MethodFlag::WRITE);
    // std::cout << nodeData.id << " -> ";
    // nodeData.id = currentID++; // Redefines the AND IDs according to the
    // topological sorting. std::cout << nodeData.id << std::endl;

    unsigned int andIndex = nodeData.id * 2;

    auto inEdge        = graph.in_edge_begin(node);
    bool lhsPolarity   = graph.getEdgeData(inEdge, galois::MethodFlag::READ);
    aig::GNode lhsNode = graph.getEdgeDst(inEdge);
    aig::NodeData& lhsNodeData =
        graph.getData(lhsNode, galois::MethodFlag::READ);
    unsigned int lhsIndex = lhsNodeData.id * 2;
    lhsIndex              = lhsPolarity ? lhsIndex : (lhsIndex + 1);

    inEdge++;
    bool rhsPolarity   = graph.getEdgeData(inEdge, galois::MethodFlag::READ);
    aig::GNode rhsNode = graph.getEdgeDst(inEdge);
    aig::NodeData& rhsNodeData =
        graph.getData(rhsNode, galois::MethodFlag::READ);
    unsigned int rhsIndex = rhsNodeData.id * 2;
    rhsIndex              = rhsPolarity ? rhsIndex : (rhsIndex + 1);

    if (lhsIndex < rhsIndex) {
      std::swap(lhsIndex, rhsIndex);
    }

    aigerFile << andIndex << " " << lhsIndex << " " << rhsIndex << std::endl;
  }
}

void AigWriter::writeAig(Aig& aig) {
  aig.resetAndIds();
  writeAigHeader(aig);
  writeLatchesAig(aig);
  writeOutputs(aig);
  writeAndsAig(aig);
  writeSymbolTable(aig);
}

void AigWriter::writeAigHeader(Aig& aig) {
  int i = aig.getNumInputs();
  int l = aig.getNumLatches();
  int o = aig.getNumOutputs();
  int a = aig.getNumAnds();
  int m = i + l + a;
  aigerFile << "aig " << m << " " << i << " " << l << " " << o << " " << a
            << std::endl;
}

void AigWriter::writeLatchesAig(Aig& aig) {

  aig::Graph& aigGraph = aig.getGraph();

  for (aig::GNode latchNode : aig.getLatchNodes()) {
    auto inEdge         = aigGraph.in_edge_begin(latchNode);
    bool inEdgePolarity = aigGraph.getEdgeData(inEdge);
    aig::GNode inNode   = aigGraph.getEdgeDst(inEdge);
    aig::NodeData& inNodeData =
        aigGraph.getData(inNode, galois::MethodFlag::READ);
    // bool initState = latchNode->getInitialValue(); // FIXME;
    if (inEdgePolarity) {
      aigerFile << inNodeData.id * 2 << std::endl;
      // aigerFile << inNodeData.id + 1 << " " << initState << std::endl;
    } else {
      aigerFile << (inNodeData.id * 2) + 1 << std::endl;
      // aigerFile << inNodeData.id + 1 << " " << initState << std::endl;
    }
  }
}

void AigWriter::writeAndsAig(Aig& aig) {

  std::stack<aig::GNode> stack;

  aig.computeTopologicalSortForAnds(stack);

  aig::Graph& graph = aig.getGraph();

  // unsigned int currentID = aig.getNumInputs() + aig.getNumLatches() + 1;

  while (!stack.empty()) {

    aig::GNode node = stack.top();
    stack.pop();

    aig::NodeData& nodeData = graph.getData(node, galois::MethodFlag::WRITE);
    // std::cout << nodeData.id << " -> ";
    // nodeData.id = currentID++; // Redefines the AND IDs according to the
    // topological sorting. std::cout << nodeData.id << std::endl;

    unsigned int andIndex = nodeData.id * 2;

    auto inEdge        = graph.in_edge_begin(node);
    bool lhsPolarity   = graph.getEdgeData(inEdge, galois::MethodFlag::READ);
    aig::GNode lhsNode = graph.getEdgeDst(inEdge);
    aig::NodeData& lhsNodeData =
        graph.getData(lhsNode, galois::MethodFlag::READ);
    unsigned int lhsIndex = lhsNodeData.id * 2;
    lhsIndex              = lhsPolarity ? lhsIndex : (lhsIndex + 1);

    inEdge++;
    bool rhsPolarity   = graph.getEdgeData(inEdge, galois::MethodFlag::READ);
    aig::GNode rhsNode = graph.getEdgeDst(inEdge);
    aig::NodeData& rhsNodeData =
        graph.getData(rhsNode, galois::MethodFlag::READ);
    unsigned int rhsIndex = rhsNodeData.id * 2;
    rhsIndex              = rhsPolarity ? rhsIndex : (rhsIndex + 1);

    if (lhsIndex < rhsIndex) {
      std::swap(lhsIndex, rhsIndex);
    }

    encode(andIndex - lhsIndex);
    encode(lhsIndex - rhsIndex);
  }
}

void AigWriter::writeSymbolTable(Aig& aig) {

  int i = 0;
  for (auto inputName : aig.getInputNames()) {
    aigerFile << "i" << i++ << " " << inputName << std::endl;
  }

  i = 0;
  for (auto latchName : aig.getLatchNames()) {
    aigerFile << "l" << i++ << " " << latchName << std::endl;
  }

  i = 0;
  for (auto outputName : aig.getOutputNames()) {
    aigerFile << "o" << i++ << " " << outputName << std::endl;
  }

  aigerFile << "c" << std::endl << aig.getDesignName() << std::endl;
}

void AigWriter::encode(unsigned x) {
  unsigned char ch;
  while (x & ~0x7f) {
    ch = (x & 0x7f) | 0x80;
    aigerFile.put(ch);
    x >>= 7;
  }
  ch = x;
  aigerFile.put(ch);
}

void AigWriter::close() { aigerFile.close(); }
