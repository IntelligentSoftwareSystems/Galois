#include "Galois/Timer.h"
#include "Galois/Galois.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Runtime/TreeExec.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

//#define PAPI
#include "simple/simplepapi.h"

#include <float.h>              // For DBL_DIG, significant digits in double

typedef galois::Graph::LC_CSR_Graph<unsigned int, double> Graph;
typedef Graph::GraphNode GNode;

typedef galois::Graph::LC_Morph_Graph<unsigned int, double> OutGraph;

Graph graph;
OutGraph outgraph;
OutGraph::GraphNode *outnodes = NULL;

static const unsigned int INVALID =
  std::numeric_limits<unsigned int>::max();

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);

struct SelfEdges {              // FIXME
  Graph::edge_iterator *selfedges;
  SelfEdges() {
    selfedges = (Graph::edge_iterator *)malloc(sizeof(Graph::edge_iterator)*graph.size());
    if ( !selfedges ) abort();
  }
  void operator()(GNode &node/*, galois::UserContext<GNode>& ctx*/) const {
    unsigned n = graph.getData(node);
    for ( Graph::edge_iterator ii = graph.edge_begin(node),
            ie = graph.edge_end(node); ii != ie; ii++ ) {
      if ( graph.getEdgeDst(ii) == node ) {
        selfedges[n] = ii;
        return;
      }
    }
    abort();
  }
  void find() {
    for ( Graph::iterator ii = graph.begin(), ie = graph.end();
          ii != ie; ii++ ) {
      GNode node = *ii;
      (*this)(node);
    }
  }
  Graph::edge_iterator get(unsigned int n) {
    return selfedges[n];
  }
};

struct ETree {
  unsigned *parent, *ancestor;
  ETree() {
    parent = (unsigned *)malloc(sizeof(unsigned)*graph.size());
    if ( !parent ) abort();
  }
  void operator()(GNode& node/*, galois::UserContext<GNode>& ctx*/) const {
    unsigned n = graph.getData(node);
    parent[n] = INVALID;
    ancestor[n] = INVALID;

    for ( Graph::edge_iterator ii = graph.edge_begin(node),
            ie = graph.edge_end(node); ii != ie; ii++ ) {
      unsigned i = graph.getData(graph.getEdgeDst(ii));
      while ( i < n && i != INVALID ) {
        unsigned nexti = ancestor[i];
        ancestor[i] = n;
        if ( nexti == INVALID )
          parent[i] = n;
        i = nexti;
      }
    }
  }

  void build() {
    ancestor = (unsigned *)malloc(sizeof(GNode)*graph.size());
    if ( !ancestor ) abort();
    for ( Graph::iterator ii = graph.begin(), ie = graph.end();
          ii != ie; ii++ ) {
      GNode node = *ii;
      (*this)(node);
    }
    free(ancestor);
    ancestor = NULL;
    /*
    printf("parent: ");
    for ( unsigned i = 0; i < graph.size(); i++ ) {
      printf("%u ", parent[i]);
    }
    printf("\n");
    */
  }

  unsigned reach(GNode &node, unsigned *stack, char *marked) {
    unsigned k = graph.getData(node);
    unsigned stackpos = graph.size();
    assert(!marked[k]);
    marked[k] = 1;

    for ( Graph::edge_iterator ii = graph.edge_begin(node),
            ie = graph.edge_end(node); ii != ie; ii++ ) {
      unsigned i = graph.getData(graph.getEdgeDst(ii));
      assert(i <= k);

      // Traverse up the elimination tree
      unsigned depth;
      for ( depth = 0; !marked[i]; i = parent[i], depth++ ) {
        stack[depth] = i;
        //printf("Found [%u]: %u\n", depth, i);
        assert(marked[i] == 0);
        marked[i] = 1;
        assert(parent[i] < graph.size());
      }

      // Move traversed elements to the stack
      // FIXME: Why is this a separate step?
      // FIXME: Why is stack from the top, not the bottom?
      while ( depth > 0 ) {
        stackpos--;
        depth--;
        stack[stackpos] = stack[depth];
        //printf("Found [%u] -> [%u]: %u == %u\n", depth, stackpos, stack[depth], stack[stackpos]);
      }
    }

    // Unmark all nodes
    for ( unsigned i = stackpos; i < graph.size(); i++ ) {
      marked[stack[i]] = 0;
    }
    marked[k] = 0;

#ifndef NDEBUG
    for ( unsigned i = 0; i < graph.size(); i++ )
      assert(marked[i] == 0);
#endif

    return stackpos;
  }
};

struct PerThread {
  double *densecol;
  unsigned *stack;
  char *etree_temp;

  PerThread() {
    densecol = (double *)malloc(graph.size()*sizeof(double));
    stack = (unsigned *)malloc(graph.size()*sizeof(unsigned));
    etree_temp = (char *)malloc(graph.size()*sizeof(char));
    if ( !densecol || !stack || !etree_temp ) abort();
    unsigned i;
    for ( i = 0; i < graph.size(); i++ ) {
      densecol[i] = 0;
      stack[i] = INVALID;
      etree_temp[i] = 0;
    }
  }
};
galois::Substrate::PerThreadStorage<PerThread> *pts = NULL;
ETree *etree = NULL;
SelfEdges *selfedges = NULL;

struct UpCholesky {
  void operator()(GNode &node/*, galois::UserContext<GNode>& ctx*/) const {
    PerThread *mypts = pts->getLocal();

    // Get self-edge
    Graph::edge_iterator node_self = selfedges->get(graph.getData(node, galois::MethodFlag::UNPROTECTED)); // FIXME
    assert(graph.getEdgeDst(node_self) == node);
    double node_self_data = graph.getEdgeData(node_self,
                                              galois::MethodFlag::UNPROTECTED);

    // Get output node
    unsigned node_id = graph.getData(node, galois::MethodFlag::UNPROTECTED);
    OutGraph::GraphNode node_out = outnodes[node_id];
    //printf("STARTING %u\n", node_id);

    // Explode dense column
    double *densecol = mypts->densecol;
    for ( Graph::edge_iterator ii = graph.edge_begin(node, galois::MethodFlag::UNPROTECTED),
            ie = graph.edge_end(node, galois::MethodFlag::UNPROTECTED); ii != ie; ii++ ) {
      // FIXME: skip self-edge
      if ( ii == node_self ) continue;
      assert(ii != node_self);
      unsigned dest = graph.getData(graph.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED);
      densecol[dest] = graph.getEdgeData(ii, galois::MethodFlag::UNPROTECTED);
    }

    // Find non-zero pattern of row
    unsigned *stack = mypts->stack;
    unsigned stackpos = etree->reach(node, stack, mypts->etree_temp);

    // Iterate over row
    for ( ; stackpos < graph.size(); stackpos++ ) {
      unsigned colnode = stack[stackpos];
      double x = densecol[colnode];
      densecol[colnode] = 0;

      //printf(" COL %u\n", colnode);
      assert(colnode != node_id);

      // Get corresponding node and self-edge from output graph
      OutGraph::GraphNode colnode_out = outnodes[colnode];
      OutGraph::edge_iterator ji = outgraph.edge_begin(colnode_out),
        je = outgraph.edge_end(colnode_out);
      assert(ji != je);
      assert(outgraph.getData(outgraph.getEdgeDst(ji), galois::MethodFlag::UNPROTECTED) == colnode);
      double col_self_data = outgraph.getEdgeData(ji, galois::MethodFlag::UNPROTECTED);
      ji++;

      // Divide by diagonal entry of the column
      assert(col_self_data != 0);
      x /= col_self_data;

      // Subtract from diagonal entry of row
      node_self_data -= x*x;

      // Do update along column
      for ( ; ji != je; ji++ ) {
        unsigned jdest = outgraph.getData(outgraph.getEdgeDst(ji), galois::MethodFlag::UNPROTECTED);
        assert(jdest != colnode);
        //printf("  %u\n", jdest);
        double jdata = outgraph.getEdgeData(ji, galois::MethodFlag::UNPROTECTED);
        double delta = x * jdata;
        densecol[jdest] -= delta;
      }

      // Insert new entry in output graph
      // colnode -> node (x)
      {
        //printf("Writing %u -> %u == %f\n", colnode, node_id, x);
        OutGraph::edge_iterator oi = outgraph.addEdge(outnodes[colnode],
                                                      node_out, galois::MethodFlag::WRITE);
        outgraph.getEdgeData(oi) = x;
      }
    }

    // Update self edge by L[i,i]=sqrt(L[i,i]-sum(L[k,i]^2, k=0..i-1))
    // (sum over outgoing edges/matrix row)
    assert(node_self_data > 0 && !isnan(node_self_data));
    node_self_data = sqrt(node_self_data);
    //*selfdata_p = selfdata; // FIXME: Remove this, but avoid perf penalty
    {
      //printf("Writing %u -> %u == %f\n", node_id, node_id, node_self_data);
      OutGraph::edge_iterator oi = outgraph.addEdge(node_out, node_out,
                                                    galois::MethodFlag::WRITE);
      outgraph.getEdgeData(oi) = node_self_data;
    }
  }
};

struct TreeExecModel {
  typedef std::vector<GNode> ChildList;
  typedef std::vector<ChildList> Children;
  Children children;
  ChildList rootnodes;

  TreeExecModel() : children(graph.size(), ChildList()) {
    Graph::iterator ii = graph.begin();
    for ( unsigned i = 0; i < graph.size(); i++, ii++ ) {
      //      assert(ii != ei);
      if ( etree->parent[i] == INVALID )
        rootnodes.push_back(*ii);
      else
        children[etree->parent[i]].push_back(*ii);
    }
  }

  struct GaloisDivide {
    TreeExecModel *tem;
    ChildList *rootnodes;
    GaloisDivide(TreeExecModel *tem, ChildList *rootnodes = NULL) :
      tem(tem), rootnodes(rootnodes) { }
    template <typename C>
    void operator () (GNode& node, C& ctx) {
      unsigned i = graph.getData(node), j;
      for ( j = 0; j < tem->children[i].size(); j++ ) {
        ctx.spawn(tem->children[i][j]);
      }
      if ( rootnodes ) {
        for ( j = 1; j < rootnodes->size(); j++ ) {
          ctx.spawn((*rootnodes)[j]);
        }
        rootnodes = NULL;
      }
    }
  };

  struct GaloisConquer {
    UpCholesky cholop;
    void operator () (GNode& node) {
      cholop(node);
    };
  };

  void run() {
    // Begin execution -- FIXME: initial elements
    printf("%u root nodes\n", rootnodes.size());
    /*
    for ( unsigned i = 0; i < rootnodes.size(); i++ ) {
      // Get pointer to given node
      galois::Runtime::for_each_ordered_tree(rootnodes[i],
                                             GaloisDivide(this),
                                             GaloisConquer(),
                                             "UpCholeskyTree");
    }
    */
    galois::Runtime::for_each_ordered_tree(rootnodes[0],
                                           GaloisDivide(this, &rootnodes),
                                           GaloisConquer(),
                                           "UpCholeskyTree");
  }
};

// include/Galois/Graphs/Serialize.h
// Output a graph to a file as an edgelist
template<typename GraphType>
bool outputTextEdgeData(const char* ofile, GraphType& G) {
  //std::ofstream file(ofile);
  FILE *file = fopen(ofile, "w");
  if ( !file ) {
    perror("fopen outfile");
    return false;
  }
  for (typename GraphType::iterator ii = G.begin(),
         ee = G.end(); ii != ee; ++ii) {
    unsigned src = G.getData(*ii);
    // FIXME: Version in include/Galois/Graphs/Serialize.h is wrong.
    for (typename GraphType::edge_iterator jj = G.edge_begin(*ii),
           ej = G.edge_end(*ii); jj != ej; ++jj) {
      unsigned dst = G.getData(G.getEdgeDst(jj));
      fprintf(file, "%d %d %.*e\n", src, dst, DBL_DIG+3, G.getEdgeData(jj));
    }
  }
  fclose(file);
  return true;
}

int main(int argc, char **argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, 0,0,0);

  galois::Graph::readGraph(graph, filename);
  {
    unsigned i = 0;
    unsigned nedges[graph.size()];
    for ( i = 0; i < graph.size(); i++ ) {
      nedges[i] = 0;
    }
    i = 0;
    for ( Graph::iterator ii = graph.begin(), ie = graph.end();
          ii != ie; ii++, i++ ) {
      graph.getData(*ii) = i;
      for ( Graph::edge_iterator ji = graph.edge_begin(*ii),
              je = graph.edge_end(*ii); ji != je; ji++ )
        nedges[graph.getData(graph.getEdgeDst(ji))]++;
    }
    outnodes = (OutGraph::GraphNode *)malloc(graph.size()*sizeof(OutGraph::GraphNode));
    if ( !outnodes ) abort();
    for ( i = 0; i < graph.size(); i++ ) {
      outnodes[i] = outgraph.createNode(nedges[i]);
      //printf("%u edges for node %u\n", nedges[i], i);
      outgraph.getData(outnodes[i]) = i;
    }
  }

  pts = new galois::Substrate::PerThreadStorage<PerThread>();
  etree = new ETree();
  selfedges = new SelfEdges();
  selfedges->find();
  etree->build();

  galois::StatTimer T("NumericTime");
  papi_start();
  T.start();
#if 0
  {
    UpCholesky cholop;
    for ( Graph::iterator ii = graph.begin(), ie = graph.end();
          ii != ie; ii++ ) {
      GNode node = *ii;
      cholop(node);
    }
  }
#else
  {
    TreeExecModel().run();
  }
#endif
  T.stop();
  papi_stop("");

  delete pts;

  outputTextEdgeData("choleskyedges.txt", outgraph);

  return 0;
}
