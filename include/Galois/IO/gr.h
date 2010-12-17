//Graph (.gr files)

/*  A graph contains n nodes and m arcs */
/*  Nodes are identified by integers 1...n */
/*  Graphs can be interpreted as directed or undirected, depending on the problem being studied */
/*  Graphs can have parallel arcs and self-loops */
/*  Arc weights are signed integers */

/* Comment lines:  Comment lines can appear anywhere and are ignored by programs. */
/* c This is a comment */
/* Problem line: The problem line is unique and must appear as the first non-comment line. This line has the format on the right, where n and m are the number of nodes and the number of arcs, respectively.	 */
/* p sp n m */
/* Arc descriptor lines: Arc descriptors are of the form on the right, where U and V are the tail and the head node ids, respectively, and W is the arc weight. */
/* a U V W */

#include <fstream>
#include <string>

namespace Galois {
namespace IO {

template<bool weighted>
struct AddEdge {
};

template<>
struct AddEdge<true> {
  template<typename GraphTy, typename GraphNodeTy>
  static void addEdge(GraphTy* graph, GraphNodeTy src, GraphNodeTy dst, int weight) {
    graph->addEdge(src,dst, weight, Galois::Graph::NONE);
  }
};

template<>
struct AddEdge<false> {
  template<typename GraphTy, typename GraphNodeTy>
  static void addEdge(GraphTy* graph, GraphNodeTy src, GraphNodeTy dst, int weight) {
    graph->addEdge(src,dst, Galois::Graph::NONE);
  }
};



template<typename GraphTy, bool weighted>
void readFile_gr(char *filename, GraphTy* graph) {
  std::ifstream infile;
  infile.open(filename, std::ifstream::in); // opens the vector file
  if (!infile) { // file couldn't be opened
    std::cerr << "Error: vector file could not be opened\n";
    exit(-1);
  }
  
  std::string name;
  int LEdgeCount = 0;
  int numNodes = 0;
  int numEdges = 0;
  typename GraphTy::GraphNode* gnodes = NULL;
  while (!infile.eof()) {
    char firstchar = 0;
    infile >> firstchar;
    if (infile.fail()) {
      break;
    } else if (firstchar == 'a') {
      ++LEdgeCount;
      int src, dest, weight;
      infile >> src >> dest >> weight;
      assert(weight > 0);
      AddEdge<weighted>::addEdge(graph, gnodes[src - 1], gnodes[dest - 1], weight);
    } else if (firstchar == 'c') {
      std::string line;
      getline(infile, line);
    } else if (firstchar == 'p') {
      infile >> name;
      infile >> numNodes;
      infile >> numEdges;
      gnodes = new typename GraphTy::GraphNode[numNodes];
      for (int i = 0; i < numNodes; i++) {
	gnodes[i] = graph->createNode(i+1);
	graph->addNode(gnodes[i], Galois::Graph::NONE);
      }
    } else {
      std::cerr << "Failure reading file " << filename
		<< " because of line begining with " << firstchar
		<< "\n";
      abort();
    } 
  }
  delete[] gnodes;
  if (LEdgeCount != numEdges) {
    std::cerr << "Read edges " << LEdgeCount << " don't match claimed " << numEdges << "\n";
    assert(LEdgeCount == numEdges);
    abort();
  }
  infile.close();
  std::cout << "Read " << numNodes << " nodes and " << numEdges << " edges.\n";
}
  
}
}
