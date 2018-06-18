//
// dag_graphs.h - Graph classes for DAG applications
//

#include <vector>
#include <algorithm>
//#include <unordered_map>
#include <boost/unordered_map.hpp>
#include <boost/dynamic_bitset.hpp>

#define INVALID ((unsigned int)-1)
typedef unsigned int node_t;
typedef unsigned int edge_t;

/* Defined in problem-specific code */

template <typename Graph>
void do_node(Graph* graph, node_t node);
void init_node(node_t id, nodedata_t* data);
void init_edge(node_t from, node_t to, edgedata_t* data, char* extra);

struct NodeData {
  NodeData() : indegree(0), outdegree(0){};
  unsigned int indegree;  /* Used for DAG scheduling */
  unsigned int outdegree; /* Used for schedule priority */
  nodedata_t data;
};

#if 0
struct GeneralGraph {
  virtual edge_t edge_begin(node_t node) const = 0;
  virtual edge_t edge_end(node_t node) const = 0;
  virtual node_t edge_dest(node_t node, edge_t edge) const = 0;
  edge_t find_edge(node_t src, node_t dest) const {
    for ( edge_t ii = edge_begin(src), ei = edge_end(src); ii != ei; ii++ ) {
      if ( edge_dest(src, ii) == dest )
        return ii;
    }
    return INVALID;
  }
};
#endif

// For searching with std::upper_bound
template <typename Edge>
bool node_lt_edge(const node_t node, const Edge& edge) {
  return node < edge.dest;
}
// For searching with std::lower_bound
template <typename Edge>
bool edge_lt_node(const Edge& edge, const node_t node) {
  return edge.dest < node;
}

template <bool SortNeighbors = false, bool IndexNeighbors = false>
struct MutableGraph {
  struct Edge {
    Edge(node_t dest, edgedata_t data) : dest(dest), data(data) {}
    Edge(node_t dest, char* extra, node_t from) : dest(dest) {
      init_edge(from, dest, &data, extra);
    }
    node_t dest;
    edgedata_t data;
  };
  typedef std::vector<Edge> EdgeList;
  struct Node {
    EdgeList neighbors;
    NodeData data;
  };
  typedef std::vector<Node> NodeList;
  NodeList nodes;

  std::vector<boost::dynamic_bitset<>> edgecheck;

  unsigned int nodecount;

  MutableGraph() {
    nodecount = 0;
    _load(stdin);
  };

  MutableGraph(FILE* fh) {
    nodecount = 0;
    _load(fh);
  };

  void _load(FILE* fh) {
    node_t from, to;
    char extra[1024] = "";

    while (fscanf(fh, "%u %u%1023[^\n]", &from, &to, extra) >= 2) {
      /* Allocate additional nodes, if necessary */
      unsigned int maxnode = to > from ? to : from;
      if (maxnode >= nodecount) {
        nodes.resize(maxnode + 1, Node());
        for (; nodecount <= maxnode; nodecount++) {
          init_node(nodecount, &nodes[nodecount].data.data); // FIXME: API
        }
      }

      /* Add the edge to the graph */
      _add_edge(from, to, extra);
#ifdef SYMMETRIC
      if (from != to && !find_edge(to, from))
        _add_edge(to, from, extra);
#endif /* SYMMETRIC */
      extra[0] = 0;
    }
    if (IndexNeighbors) {
      edgecheck.insert(edgecheck.end(), nodecount,
                       boost::dynamic_bitset<>(nodecount));
      for (node_t node = 0; node < nodecount; node++) {
        for (edge_t ii = edge_begin(node), ei = edge_end(node); ii < ei; ii++) {
          node_t dest           = edge_dest(node, ii);
          edgecheck[node][dest] = 1;
        }
      }
    }
    printf("Read %u nodes\n", nodecount);
  }

  void add_edge(node_t from, node_t to, edgedata_t data) {
    Edge edge(to, data);
    _add_edge(from, to, edge);
  }

  void _add_edge(node_t from, node_t to, char* extra) {
    Edge edge(to, extra, from);
    _add_edge(from, to, edge);
  }

  void _add_edge(node_t from, node_t to, Edge& edge) {
    typename EdgeList::iterator ii = nodes[from].neighbors.end();
    if (SortNeighbors)
      ii = std::upper_bound(nodes[from].neighbors.begin(), ii, to,
                            node_lt_edge<Edge>);
    // nodes[from].neighbors.push_back(edge);
    nodes[from].neighbors.insert(ii, edge);
    nodes[from].data.outdegree++;
    if (to != from)
      nodes[to].data.indegree++;
    if (IndexNeighbors && edgecheck.size() > 0)
      edgecheck[from][to] = 1;
  }

  nodedata_t* node_data(node_t node) { return &nodes[node].data.data; }

  NodeData& _node_data(node_t node) { return nodes[node].data; }

  edge_t edge_begin(node_t node) const { return 0; }

  edge_t edge_end(node_t node) const { return nodes[node].neighbors.size(); }

  edgedata_t* edge_data(node_t node, edge_t edge) {
    Node& node_obj = nodes[node];
    Edge& edge_obj = node_obj.neighbors[edge];
    return &edge_obj.data;
  }
  node_t edge_dest(node_t node, edge_t edge) const {
    const Node& node_obj = nodes[node];
    const Edge& edge_obj = node_obj.neighbors[edge];
    return edge_obj.dest;
  }

  edge_t find_edge(node_t src, node_t dest) const {
    if (IndexNeighbors && !edgecheck[src][dest])
      return INVALID;
    if (SortNeighbors) {
      typename EdgeList::const_iterator ii = nodes[src].neighbors.begin(),
                                        ei = nodes[src].neighbors.end(),
                                        ci = std::lower_bound(
                                            ii, ei, dest, edge_lt_node<Edge>);
      return (ci == ei || ci->dest != dest) ? INVALID : ci - ii;
    }
    for (edge_t ii = edge_begin(src), ei = edge_end(src); ii != ei; ii++) {
      if (edge_dest(src, ii) == dest)
        return ii;
    }
    return INVALID;
  }
};

struct CRSGraph {
  NodeData* nodedata;
  edge_t* edgeidx;

  node_t* edgedest;
  edgedata_t* edgedata;

  unsigned int nodecount;
  int SortNeighbors;

  template <typename Graph>
  CRSGraph(Graph& orig) {
    nodedata = (NodeData*)malloc(orig.nodecount * sizeof(NodeData));
    edgeidx  = (edge_t*)malloc(orig.nodecount * sizeof(edge_t));
    if (!nodedata || !edgeidx)
      abort();
    // Fill node data, count number of edges
    unsigned int edgecount = 0;
    nodecount              = 0;
    for (node_t node = 0; node < orig.nodecount; node++) {
      memcpy(&nodedata[nodecount], &orig._node_data(node), sizeof(NodeData));
      for (edge_t ii = orig.edge_begin(node), ei = orig.edge_end(node);
           ii != ei; ii++) {
        edgecount++;
      }
      edgeidx[nodecount] = edgecount;
      nodecount++;
    }
    assert(nodecount == orig.nodecount);
    edgedest = (node_t*)malloc(edgecount * sizeof(node_t));
    edgedata = (edgedata_t*)malloc(edgecount * sizeof(edgedata_t));
    if (!edgedest || !edgedata)
      abort();
    edgecount = 0;
    for (node_t node = 0; node < nodecount; node++) {
      for (edge_t ii = orig.edge_begin(node), ei = orig.edge_end(node);
           ii != ei; ii++) {
        edgedest[edgecount] = orig.edge_dest(node, ii);
        memcpy(&edgedata[edgecount], orig.edge_data(node, ii),
               sizeof(edgedata_t));
        edgecount++;
      }
    }
    assert(edgecount == edgeidx[nodecount - 1]);
    // Check for sorted neighbors
    SortNeighbors = 1;
    for (node_t node = 0; node < nodecount; node++) {
      node_t prev = INVALID;
      for (edge_t ii = edge_begin(node), ei = edge_end(node); ii != ei; ii++) {
        node_t dest = edge_dest(node, ii);
        if (prev != INVALID && prev > dest) {
          SortNeighbors = 0;
          break;
        }
        prev = dest;
      }
      if (!SortNeighbors)
        break;
    }
    printf("SortNeighbors: %d\n", SortNeighbors);
  }

  nodedata_t* node_data(node_t node) { return &nodedata[node].data; }

  NodeData& _node_data(node_t node) { return nodedata[node]; }

  edge_t edge_begin(node_t node) const {
    return node > 0 ? edgeidx[node - 1] : 0;
  }

  edge_t edge_end(node_t node) const { return edgeidx[node]; }

  edgedata_t* edge_data(node_t node, edge_t edge) { return &edgedata[edge]; }
  node_t edge_dest(node_t node, edge_t edge) const { return edgedest[edge]; }

  edge_t find_edge(node_t src, node_t dest) const {
    edge_t ii = edge_begin(src), ei = edge_end(src);
    if (SortNeighbors) {
      while (ii < ei) {
        edge_t ix = (ii + ei) / 2;
        assert(ix < ei);
        node_t idest = edge_dest(src, ix);
        if (idest < dest)
          ii = ix + 1;
        else if (idest > dest)
          ei = ix;
        else
          return ix;
      }
    } else {
      for (; ii != ei; ii++) {
        if (edge_dest(src, ii) == dest)
          return ii;
      }
    }
    return INVALID;
  }
};

struct BidiGraph : CRSGraph {
  edge_t* inedgeidx;

  node_t* inedgesrc;
  unsigned* inedgedataidx;

  template <typename Graph>
  BidiGraph(Graph& orig) : CRSGraph(orig) {
    if (!SortNeighbors)
      abort();
    unsigned edgecount = edgeidx[nodecount - 1];
    inedgeidx          = (edge_t*)malloc(nodecount * sizeof(edge_t));
    inedgesrc          = (node_t*)malloc(edgecount * sizeof(node_t));
    inedgedataidx      = (unsigned*)malloc(edgecount * sizeof(unsigned));
    unsigned temp[nodecount];
    // Count incoming edges for each node
    for (unsigned i = 0; i < nodecount; i++)
      temp[i] = 0;
    for (node_t src = 0; src < nodecount; src++) {
      for (edge_t ii = edge_begin(src), ei = edge_end(src); ii != ei; ii++) {
        node_t dest = edge_dest(src, ii);
        temp[dest]++;
      }
    }
    // Store cumulative sums as inedgeidx (compressed storage)
    for (unsigned i = 0; i < nodecount; i++) {
      unsigned start = i > 0 ? inedgeidx[i - 1] : 0;
      inedgeidx[i]   = temp[i] + start;
      temp[i]        = start;
    }
    // Insert edges into list
    for (node_t src = 0; src < nodecount; src++) {
      for (edge_t ii = edge_begin(src), ei = edge_end(src); ii != ei; ii++) {
        node_t dest               = edge_dest(src, ii);
        inedgesrc[temp[dest]]     = src;
        inedgedataidx[temp[dest]] = ii;
        temp[dest]++;
      }
    }
    // Sort incoming edges (linear time sort using temp)
    for (unsigned i = 0; i < nodecount; i++)
      temp[i] = INVALID;
    for (node_t dest = 0; dest < nodecount; dest++) {
      // Explode inedges data into temp
      for (edge_t ii = inedge_begin(dest), ei = inedge_end(dest); ii != ei;
           ii++) {
        temp[inedge_src(dest, ii)] = inedgedataidx[ii];
      }
      // Recompress inedges in sorted order
      unsigned idx = inedge_begin(dest);
      for (unsigned j = 0; j < nodecount; j++) {
        if (temp[j] != INVALID) {
          inedgesrc[idx]     = j;
          inedgedataidx[idx] = temp[j];
          temp[j]            = INVALID;
          idx++;
        }
      }
      assert(idx == inedge_end(dest));
    }
  }

  edge_t inedge_begin(node_t node) const {
    return node > 0 ? inedgeidx[node - 1] : 0;
  }

  edge_t inedge_end(node_t node) const { return inedgeidx[node]; }

  edgedata_t* inedge_data(node_t node, edge_t edge) {
    return &edgedata[inedgedataidx[edge]];
  }
  node_t inedge_src(node_t node, edge_t edge) const { return inedgesrc[edge]; }
};

struct MapGraph {
  std::vector<NodeData> nodedata;
  std::vector<edge_t> edgeidx;

  std::vector<node_t> edgedest;
  std::vector<edgedata_t> edgedata;

  std::vector<boost::unordered_map<node_t, edge_t>> edgesearch;
  std::vector<boost::dynamic_bitset<>> edgecheck;
  // std::vector<edge_t> selfedge;

  unsigned int nodecount;

  template <typename Graph>
  MapGraph(Graph& orig) {
    nodecount = orig.nodecount;
    nodedata.reserve(nodecount);
    edgeidx.reserve(nodecount);
    // Fill node data, count number of edges
    unsigned int edgecount = 0;
    for (node_t node = 0; node < nodecount; node++) {
      nodedata.push_back(orig._node_data(node));
      for (edge_t ii = orig.edge_begin(node), ei = orig.edge_end(node);
           ii != ei; ii++) {
        edgedest.push_back(orig.edge_dest(node, ii));
        edgedata.push_back(*orig.edge_data(node, ii));
        edgecount++;
      }
      edgeidx.push_back(edgecount);
    }
    edgedest.resize(edgecount);
    edgedata.resize(edgecount);
    // Fill edgesearch
    edgesearch.resize(nodecount);
    // selfedge.resize(nodecount);
    edgecheck.insert(edgecheck.end(), nodecount,
                     boost::dynamic_bitset<>(nodecount));
    for (node_t node = 0; node < nodecount; node++) {
      // unsigned nedges = 0;
      // selfedge[node] = INVALID;
      for (edge_t ii = edge_begin(node), ei = edge_end(node); ii < ei; ii++) {
        node_t dest            = edge_dest(node, ii);
        edgesearch[node][dest] = ii;
        edgecheck[node][dest]  = 1;
        // if ( node == dest )
        //  selfedge[node] = ii;
        // nedges++;
      }
      // edgesearch[node].rehash(nedges*5);
    }
  }

  nodedata_t* node_data(node_t node) { return &nodedata[node].data; }

  NodeData& _node_data(node_t node) { return nodedata[node]; }

  edge_t edge_begin(node_t node) const {
    return node > 0 ? edgeidx[node - 1] : 0;
  }

  edge_t edge_end(node_t node) const { return edgeidx[node]; }

  edgedata_t* edge_data(node_t node, edge_t edge) { return &edgedata[edge]; }
  node_t edge_dest(node_t node, edge_t edge) const { return edgedest[edge]; }

  edge_t find_edge(node_t src, node_t dest) const {
    // if ( src == dest )
    //  return selfedge[src];
    if (!edgecheck[src][dest])
      return INVALID;
    auto& the_map = edgesearch[src];
    auto ii       = the_map.find(dest);
    if (ii == edgesearch[src].end())
      return INVALID;
    else
      return ii->second;
  }
};
