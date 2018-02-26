#include "galois/graphs/FileGraph.h"
#include "galois/gIO.h"
#include "llvm/Support/CommandLine.h"

#include <vector>
#include <iostream>
#include <fstream>

//File 1 format V1:
//version (2) {uint64_t LE}
//numNodes {uint64_t LE}
//numEdges {uint64_t LE}
//inindexs[numNodes+1] {uint32_t LE} 
//potential padding (32bit max) to Re-Align to 64bits
//inedges[numEdges] {uint32_t LE}
//potential padding (32bit max) to Re-Align to 64bits
//outindexs[numNodes+1] {uint32_t LE} 
//potential padding (32bit max) to Re-Align to 64bits
//
//File 2 format V1:
//version (2) {uint64_t LE}
//outedges[numEdges] {uint32_t LE}
//potential padding (32bit max) to Re-Align to 64bits

namespace cll = llvm::cl;

static cll::opt<std::string> inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional, cll::desc("<output base filename>"), cll::Required);
static cll::opt<std::string> transposeGraphName("graphTranspose", cll::desc("Transpose of input graph"));

typedef galois::graphs::FileGraph Graph;
typedef galois::graphs::FileGraph::GraphNode GNode;

struct GraphArrays {
  std::vector<uint32_t> inIdx;
  std::vector<uint32_t> ins;
  std::vector<uint32_t> outIdx;
  std::vector<uint32_t> outs;
  uint64_t numNodes;
  uint64_t numEdges;

  void init(uint64_t numNodes, uint64_t numEdges) {
    this->numNodes = numNodes;
    this->numEdges = numEdges;
    inIdx.resize(numNodes + 1);
    ins.resize(numEdges);
    outIdx.resize(numNodes + 1);
    outs.resize(numEdges);
  }
};

void writeArray32(std::ofstream& out, std::vector<uint32_t>& x) {
  uint32_t padding = 0;
  out.write((char*) &x[0], sizeof(x[0]) * x.size());
  if (x.size() % 2)
    out.write((char*) &padding, sizeof(padding));
}

void writeToFile(GraphArrays& a, std::string f1, std::string f2) {
  uint64_t magic = 2;

  std::cout << "Writing " << a.numNodes << " nodes and " << a.numEdges << " edges\n";

  std::ofstream fout(f1.c_str());
  fout.write((char*) &magic, sizeof(magic));
  fout.write((char*) &a.numNodes, sizeof(a.numNodes));
  fout.write((char*) &a.numEdges, sizeof(a.numEdges));
  writeArray32(fout, a.inIdx);
  writeArray32(fout, a.ins);
  writeArray32(fout, a.outIdx);
  fout.close();

  std::ofstream fout2(f2.c_str());
  fout2.write((char*) &magic, sizeof(magic));
  writeArray32(fout2, a.outs);
  fout2.close();
}

ptrdiff_t findEdge(Graph& graph, GNode n, GNode t) {
  for (Graph::edge_iterator ii = graph.edge_begin(n), ei = graph.edge_end(n); ii != ei; ++ii) {
    GNode src = graph.getEdgeDst(ii);
    if (src == t) {
      return std::distance(graph.edge_data_begin<int>(), &graph.getEdgeData<int>(ii));
    }
  }
  return ~0;
}

void constructFromGraph(Graph& graph, Graph* transpose, GraphArrays& a) {
  a.init(graph.size(), graph.sizeEdges());

  uint64_t curOutIdx = 0;
  uint64_t curOutEdge = 0;
  a.outIdx[curOutIdx++] = 0;
  for (Graph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode n = *ii;
    for (Graph::edge_iterator jj = graph.edge_begin(n), ej = graph.edge_end(n); jj != ej; ++jj) {
      GNode dst = graph.getEdgeDst(jj);
      assert(curOutEdge < a.numEdges);
      a.outs[curOutEdge++] = dst;
    }
    assert(curOutIdx < a.numNodes + 1);
    a.outIdx[curOutIdx++] = curOutEdge;
  }

  uint64_t curInEdge = 0;
  uint64_t curInIdx = 0;
  a.inIdx[curInIdx++] = 0;
  for (Graph::iterator ii = transpose->begin(), ei = transpose->end(); ii != ei; ++ii) {
    GNode n = *ii;
    for (Graph::edge_iterator jj = transpose->edge_begin(n), ej = transpose->edge_end(n); jj != ej; ++jj) {
      GNode dst = transpose->getEdgeDst(jj);
      ptrdiff_t edgeIdx = findEdge(graph, dst, n);
      if (edgeIdx == (ptrdiff_t)~0) 
        GALOIS_DIE("Graph not symmetric");
      assert(curInEdge < a.numEdges);
      a.ins[curInEdge++] = edgeIdx;
    }
    assert(curInIdx < a.numNodes + 1);
    a.inIdx[curInIdx++] = curInEdge;
  }
}

void readGraph(Graph& g, const std::string& name) {
  g.fromFile(name);
  std::cout << "Read " << g.size() << " nodes and " << g.sizeEdges() << " edges\n";
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  llvm::cl::ParseCommandLineOptions(argc, argv);

  Graph g;
  GraphArrays arrays;
  std::string outFilename1 = outputFilename + "1.gr";
  std::string outFilename2 = outputFilename + "2.gr";

  readGraph(g, inputFilename);
  if (transposeGraphName == "") {
    constructFromGraph(g, &g, arrays);
  } else {
    Graph transpose;
    readGraph(transpose, transposeGraphName);
    constructFromGraph(g, &transpose, arrays);
  }
  writeToFile(arrays, outFilename1, outFilename2);

  return 0;
}
