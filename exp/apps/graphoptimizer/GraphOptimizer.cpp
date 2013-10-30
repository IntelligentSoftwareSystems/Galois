/** Graph Optimizer -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
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
 * Test the effects of edge compression on a graph
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Statistic.h"
#include "Galois/Galois.h"
#include "Galois/Graph/LCGraph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <fstream>

template<typename Iter>
void deltaCode(Iter b, Iter e) {
  if (b == e) return;
  auto val = *b++;
  while (b != e) {
    auto tmp = *b;
    assert (tmp >= val);
    *b = tmp - val;
    val = tmp;
    ++b;
  }
}

template<typename Iter>
std::deque<uint8_t> varCodeBW(Iter b, Iter e) {
  std::deque<uint8_t> retval;
  while (b != e) {
    uint32_t tmp = *b++;
    int numbits = 32 - __builtin_clz(tmp|1);
    //    auto oldsz = retval.size();
    if (numbits <= 7) {
      assert((0x7F & tmp) == tmp);
      retval.push_back(0x80 | tmp);
    } else if (numbits <= 14) {
      assert((0x3FFF & tmp) == tmp);
      retval.push_back(0x80 | (tmp >> 7));
      retval.push_back((tmp & 0x7F));
    } else if (numbits <= 21) {
      assert((0x1FFFFF & tmp) == tmp);
      retval.push_back(0x80 | (tmp >> 14));
      retval.push_back((tmp >> 7) & 0x7F);
      retval.push_back((tmp & 0x7F));
    } else if (numbits <= 28) {
      assert((0x0FFFFFFF & tmp) == tmp);
      retval.push_back(0x80 | (tmp >> 21));
      retval.push_back((tmp >> 14) & 0x7F);
      retval.push_back((tmp >>  7) & 0x7F);
      retval.push_back((tmp & 0x7F));
    } else {
      retval.push_back(0x80 | (tmp >> 28));
      retval.push_back((tmp >> 21) & 0x7F);
      retval.push_back((tmp >> 14) & 0x7F);
      retval.push_back((tmp >>  7) & 0x7F);
      retval.push_back((tmp & 0x7F));
    }
    // auto sz = retval.size();
    // auto oldsz2 = oldsz;
    // printf("0x%x %d 0x", tmp, numbits);
    // while (oldsz < sz)
    //   printf("%x", retval[oldsz++]);
    // printf("\t");

    // for (int i = 31; i >= 0; --i)
    //   printf("%d%s", ((tmp >> i) & 1), (i % 7) == 0 ? " " : "" );
    // printf("\t");
    // while (oldsz2 < sz) {
    //   printf(" ");
    //   uint8_t v = retval[oldsz2++];
    //   for (int i = 7; i >= 0; --i)
    //     printf("%d", ((v >> i) & 1) );
    // }
    // printf("\n");

  }
  return retval;
}

std::pair<uint32_t, uint8_t*> decodeOne(uint8_t* b, uint32_t state) {
  uint32_t retval = *b++;
  retval &= 0x7F;
  uint8_t rd = *b;
  while (!(rd & 0x80)) {
    b++;
    retval <<= 7;
    retval |= rd;
    rd = *b;
  }
  return std::make_pair(state+retval, b);
}

namespace Galois {
namespace Graph {
template<typename NodeTy, typename EdgeTy,
  bool HasNoLockable=false,
  bool UseNumaAlloc=false,
  bool HasOutOfLineLockable=false>
class LC_CCSR_Graph:
    private boost::noncopyable,
    private detail::LocalIteratorFeature<UseNumaAlloc>,
    private detail::OutOfLineLockableFeature<HasOutOfLineLockable && !HasNoLockable> {
  template<typename Graph> friend class LC_InOut_Graph;

public:
  template<bool _has_id>
  struct with_id { typedef LC_CCSR_Graph type; };

  template<typename _node_data>
  struct with_node_data { typedef LC_CCSR_Graph<_node_data,EdgeTy,HasNoLockable,UseNumaAlloc,HasOutOfLineLockable> type; };

  //! If true, do not use abstract locks in graph
  template<bool _has_no_lockable>
  struct with_no_lockable { typedef LC_CCSR_Graph<NodeTy,EdgeTy,_has_no_lockable,UseNumaAlloc,HasOutOfLineLockable> type; };

  //! If true, use NUMA-aware graph allocation
  template<bool _use_numa_alloc>
  struct with_numa_alloc { typedef LC_CCSR_Graph<NodeTy,EdgeTy,HasNoLockable,_use_numa_alloc,HasOutOfLineLockable> type; };

  //! If true, store abstract locks separate from nodes
  template<bool _has_out_of_line_lockable>
  struct with_out_of_line_lockable { typedef LC_CCSR_Graph<NodeTy,EdgeTy,HasNoLockable,UseNumaAlloc,_has_out_of_line_lockable> type; };

  typedef read_default_graph_tag read_tag;

protected:
  typedef LargeArray<EdgeTy> EdgeData;
  typedef LargeArray<uint8_t> EdgeDst;
  typedef detail::NodeInfoBase<NodeTy,!HasNoLockable && !HasOutOfLineLockable> NodeInfo;
  typedef LargeArray<uint64_t> EdgeIndData;
  typedef LargeArray<NodeInfo> NodeData;

public:
  typedef uint32_t GraphNode;
  typedef EdgeTy edge_data_type;
  typedef NodeTy node_data_type;
  typedef typename EdgeData::reference edge_data_reference;
  typedef typename NodeInfo::reference node_data_reference;
  struct edge_iterator {
    uint8_t* base;
    uint32_t state;
    edge_iterator& operator++() { 
      auto tmp = decodeOne(base, state);
      base = tmp.second;
      state = tmp.first;
      return *this;
    }

    bool operator==(edge_iterator& rhs) const {
      return base == rhs.base;
    }
    bool operator!=(edge_iterator& rhs) const {
      return base != rhs.base;
    }

    constexpr edge_iterator(uint8_t* b = nullptr) :base(b), state(0) {}
  };
  typedef boost::counting_iterator<unsigned> iterator;
  typedef iterator const_iterator;
  typedef iterator local_iterator;
  typedef iterator const_local_iterator;

protected:
  NodeData nodeData;
  EdgeIndData edgeIndData;
  EdgeDst edgeDst;
  EdgeData edgeData;

  uint64_t numNodes;
  uint64_t numEdges;

  typedef detail::EdgeSortIterator<GraphNode,typename EdgeIndData::value_type,EdgeDst,EdgeData> edge_sort_iterator;

  edge_iterator raw_begin(GraphNode N)  {
    return edge_iterator((N == 0) ? &edgeDst[0] : &edgeDst[edgeIndData[N-1]]);
  }

  edge_iterator raw_end(GraphNode N)  {
    return edge_iterator(&edgeDst[edgeIndData[N]]);
  }

  template<bool _A1 = HasNoLockable, bool _A2 = HasOutOfLineLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<!_A1 && !_A2>::type* = 0) {
    Galois::Runtime::acquire(&nodeData[N], mflag);
  }

  template<bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A1 && !_A2>::type* = 0) {
    this->outOfLineAcquire(getId(N), mflag);
  }

  template<bool _A1 = HasOutOfLineLockable, bool _A2 = HasNoLockable>
  void acquireNode(GraphNode N, MethodFlag mflag, typename std::enable_if<_A2>::type* = 0) { }

  size_t getId(GraphNode N) {
    return N;
  }

  GraphNode getNode(size_t n) {
    return n;
  }

public:
  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    Galois::Runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
    return NI.getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) {
    Galois::Runtime::checkWrite(mflag, false);
    return edgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return decodeOne(ni.base, ni.state).first;
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }

  const_local_iterator local_begin() const { return const_local_iterator(this->localBegin(numNodes)); }
  const_local_iterator local_end() const { return const_local_iterator(this->localEnd(numNodes)); }
  local_iterator local_begin() { return local_iterator(this->localBegin(numNodes)); }
  local_iterator local_end() { return local_iterator(this->localEnd(numNodes)); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    if (Galois::Runtime::shouldLock(mflag)) {
      for (edge_iterator ii = raw_begin(N), ee = raw_end(N); ii != ee; ++ii) {
        acquireNode(decodeOne(ii.base, ii.state).first, mflag);
      }
    }
    return raw_begin(N);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    acquireNode(N, mflag);
    return raw_end(N);
  }

  detail::EdgesIterator<LC_CCSR_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
    return detail::EdgesIterator<LC_CCSR_Graph>(*this, N, mflag);
  }

  void allocateFrom(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    if (UseNumaAlloc) {
      nodeData.allocateLocal(numNodes, false);
      edgeIndData.allocateLocal(numNodes, false);
      edgeDst.allocateLocal(numEdges*sizeof(uint32_t), false);
      edgeData.allocateLocal(numEdges, false);
      this->outOfLineAllocateLocal(numNodes, false);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges * sizeof(uint32_t));
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructFrom(FileGraph& graph, unsigned tid, unsigned total) {
    auto r = graph.divideBy(
        NodeData::sizeof_value + EdgeIndData::sizeof_value + LC_CCSR_Graph::sizeof_out_of_line_value,
        EdgeDst::sizeof_value + EdgeData::sizeof_value,
        tid, total);
    this->setLocalRange(*r.first, *r.second);
    if (tid == 0) {
    uint64_t offset = 0;
    for (FileGraph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
      nodeData.constructAt(*ii);
      this->outOfLineConstructAt(*ii);
      std::vector<uint32_t> dsts;
      for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
        if (EdgeData::has_value)
          edgeData.set(*nn, graph.getEdgeData<typename EdgeData::value_type>(nn));
        dsts.push_back(graph.getEdgeDst(nn));
      }
      std::sort(dsts.begin(), dsts.end());
      deltaCode(dsts.begin(), dsts.end());
      std::deque<uint8_t> bw = varCodeBW(dsts.begin(), dsts.end());
      std::copy(bw.begin(), bw.end(), &edgeDst[offset]);
      offset += bw.size();
      edgeIndData[*ii] = offset;
      // auto foo = decodeOne(&*bw.begin(), 0);
      // std::cout << "\n" << *dsts.begin() << " " << foo.first << " " << (foo.second - &*bw.begin()) << "\n\n";
    }
    edgeDst[offset] = 0x80;
    }
  }
};

} // end namespace
} // end namespace


typedef Galois::Graph::LC_CSR_Graph<unsigned, void> Graph;
typedef Galois::Graph::LC_CCSR_Graph<unsigned, void> GraphC;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outfilename(cll::Positional, cll::desc("<output file>"), cll::Optional);
static cll::opt<unsigned> sourcearg("source", cll::init(0), cll::desc("source for bfs"), cll::Optional);
bool dostat;

std::vector<unsigned int> raw, delta;
std::vector<unsigned int> lenBW;

unsigned long int total_elem = 0;
unsigned long int total_bytesBW = 0;

Graph graph;
GraphC graphc;


template<typename Gr>
struct AsyncBFS {
  
  typedef typename Gr::GraphNode GNode;
  typedef std::pair<GNode, unsigned> WorkItem;

  struct Indexer: public std::unary_function<WorkItem,unsigned> {
    unsigned operator()(const WorkItem& val) const {
      return val.second;
    }
  };

  struct Process {
    typedef int tt_does_not_need_aborts;

    Gr& gr;
    Process(Gr& g): gr(g) { }

    void operator()(WorkItem& item, Galois::UserContext<WorkItem>& ctx) const {
      GNode n = item.first;

      unsigned newDist = item.second;
      if (newDist > gr.getData(n, Galois::MethodFlag::NONE))
        return;

      ++newDist;

      for (typename Gr::edge_iterator ii = gr.edge_begin(n, Galois::MethodFlag::NONE),
             ei = gr.edge_end(n, Galois::MethodFlag::NONE); ii != ei; ++ii) {
        GNode dst = gr.getEdgeDst(ii);
        volatile unsigned* ddata = &gr.getData(dst, Galois::MethodFlag::NONE);
        
        unsigned  oldDist;
        while (true) {
          oldDist = *ddata;
          if (oldDist <= newDist)
            break;
          if (__sync_bool_compare_and_swap(ddata, oldDist, newDist)) {
            ctx.push(WorkItem(dst, newDist));
            break;
          }
        }
      }
    }
  };

  void operator()(Gr& graph, const GNode& source, const char* name) const {
    using namespace Galois::WorkList;
    typedef dChunkedFIFO<64> dChunk;
    //typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<Indexer,dChunk> OBIM;
    
    Galois::do_all_local(graph, [&graph] (const GNode& n) { graph.getData(n) = ~0; }, Galois::loopname("init"));

    graph.getData(source) = 0;

    Galois::for_each(WorkItem(source, 0), Process(graph), Galois::wl<OBIM>(), Galois::loopname(name));
  }
};

template<typename Iter>
void hist(Iter b, Iter e, std::vector<unsigned int>& hvals) {
  while (b != e)
    __sync_fetch_and_add(&hvals[(*b++)/1024], 1);
}

void dumphist(std::ostream& of, std::string name, std::vector<unsigned int>& hvals) {
  for (unsigned x = 0; x < hvals.size(); ++x)
    if (hvals[x])
      of << name << "," << x << "," << hvals[x] << "\n";
}





struct ComputeRatio {
  template<typename GNode>
  void operator()(const GNode& n) {
    std::deque<unsigned int> IDs;
    std::deque<unsigned char> var;
    for (Graph::edge_iterator ii = graph.edge_begin(n),
           ee = graph.edge_end(n); ii != ee; ++ii)
      IDs.push_back(graph.getEdgeDst(ii));
    std::sort(IDs.begin(), IDs.end());
    if (dostat)
      hist(IDs.begin(), IDs.end(), raw);
    deltaCode(IDs.begin(), IDs.end());
    if (dostat)
      hist(IDs.begin(), IDs.end(), delta);

    __sync_fetch_and_add(&total_elem, IDs.size());
    var = varCodeBW(IDs.begin(), IDs.end());
    __sync_fetch_and_add(&total_bytesBW, var.size());
  }
};

struct HNode {
  HNode* left;
  HNode* right;
  uint32_t val;
  uint64_t freq;
};

struct gtHNodePtr {
  bool operator()(const HNode* lhs, const HNode* rhs) const {
    return lhs->freq > rhs->freq;
  }
};


struct codeEntry {
  uint32_t val;
  unsigned nbits;
  uint32_t code;
};

struct ltcodeEntry {
  bool operator()(const codeEntry& lhs, const codeEntry& rhs) {
    return lhs.val < rhs.val;
  }
  bool operator()(const codeEntry& lhs, uint32_t val) {
    return lhs.val < val;
  }
};

HNode* buildTree(std::deque<HNode>& symbols) {
  std::priority_queue<HNode*, std::deque<HNode*>, gtHNodePtr> Q;
  for(auto ii = symbols.begin(), ee = symbols.end(); ii != ee; ++ii)
    Q.push(&*ii);

  while (Q.size() > 1) {
    HNode* n1 = Q.top(); Q.pop();
    HNode* n2 = Q.top(); Q.pop();
    symbols.push_back({n1,n2,0,n1->freq + n2->freq});
    Q.push(&symbols.back());
  }
  return Q.top();
}

void buildTableInternal(codeEntry e, HNode* n, std::deque<codeEntry>& codes) {
  if (n->left) {
    buildTableInternal({0, e.nbits + 1, e.code << 1}, n->left, codes);
    buildTableInternal({0, e.nbits + 1, 1 | (e.code << 1)}, n->right, codes);
  } else {
    e.val = n->val;
    codes.push_back(e);
  }
}

std::deque<codeEntry> buildTable(HNode* root) {
  std::deque<codeEntry> retval;
  buildTableInternal({0,0,0}, root, retval);
  return retval;
}

std::pair<unsigned long, unsigned> tryHuff() {
  std::map<uint32_t, HNode*> hyst;
  std::deque<HNode> symbols;
  std::deque<uint32_t> data;

  for (auto ni = graph.begin(), ne = graph.end(); ni != ne; ++ni) {
    std::deque<uint32_t> local;
    for (auto ei = graph.edge_begin(*ni), ee = graph.edge_end(*ni); ei != ee; ++ei) {
      local.push_back(graph.getEdgeDst(ei));
    }
    std::sort(local.begin(), local.end());
    deltaCode(local.begin(), local.end());
    std::copy(local.begin(), local.end(), std::back_inserter(data));
    for (auto ii = local.begin(), ee = local.end(); ii != ee; ++ii) {
      HNode*& n = hyst[*ii];
      if (!n) {
        symbols.push_back({nullptr, nullptr, *ii, 0});
        n = &symbols.back();
      }
      n->freq++;
    }
  }

  HNode* root = buildTree(symbols);
  std::cout << "tree built\n";

  std::deque<codeEntry> table = buildTable(root);
  std::sort(table.begin(), table.end(), ltcodeEntry());
  std::cout << "table built\n";

  unsigned long total = 0;
  for (auto ii = hyst.begin(), ee = hyst.end(); ii != ee; ++ii)
    total += std::lower_bound(table.begin(), table.end(), ii->first, ltcodeEntry())->nbits * ii->second->freq;

  return std::make_pair(total, table.size());
}

std::pair<unsigned long, unsigned> tryHuffDeltaOnly() {
  std::map<uint32_t, HNode*> hyst;
  std::deque<HNode> symbols;
  std::deque<uint32_t> data;
  unsigned long total = 0;

  for (auto ni = graph.begin(), ne = graph.end(); ni != ne; ++ni) {
    std::deque<uint32_t> local;
    for (auto ei = graph.edge_begin(*ni), ee = graph.edge_end(*ni); ei != ee; ++ei) {
      local.push_back(graph.getEdgeDst(ei));
    }
    std::sort(local.begin(), local.end());
    deltaCode(local.begin(), local.end());
    std::copy(local.begin() + 1, local.end(), std::back_inserter(data));
    total += 4 * 8;
    for (auto ii = local.begin() + 1, ee = local.end(); ii != ee; ++ii) {
      HNode*& n = hyst[*ii];
      if (!n) {
        symbols.push_back({nullptr, nullptr, *ii, 0});
        n = &symbols.back();
      }
      n->freq++;
    }
  }

  HNode* root = buildTree(symbols);
  std::cout << "tree built\n";
    
  std::deque<codeEntry> table = buildTable(root);
  std::sort(table.begin(), table.end(), ltcodeEntry());
  std::cout << "table built\n";

  for (auto ii = hyst.begin(), ee = hyst.end(); ii != ee; ++ii)
    total += std::lower_bound(table.begin(), table.end(), ii->first, ltcodeEntry())->nbits * ii->second->freq;

  return std::make_pair(total, table.size());
}


int main(int argc, char **argv) {
  Galois::StatManager statManager;
  LonestarStart(argc, argv, 0,0,0);

  dostat = outfilename.size() > 1;
  if (dostat)
    std::cout << "Collecting All Histograms\n";

  Galois::Graph::readGraph(graph, filename);
  Galois::Graph::readGraph(graphc, filename);

  for (unsigned int x = 0; x < 0; ++x) {
    auto ii = graph.edge_begin(x);
    auto iic = graphc.edge_begin(x);
    auto ee = graph.edge_end(x);
    auto eec = graphc.edge_end(x);
    int count = 0;
    while (ii != ee && iic != eec) {
      if (graph.getEdgeDst(ii) != graphc.getEdgeDst(iic)) {
        std::cout << "Mismatch at " << x << "," << count << " : " << graph.getEdgeDst(ii) << " " <<  graphc.getEdgeDst(iic) << "\n";
      }
      ++count;
      ++ii;
      ++iic;
    }
    if (ii != ee) 
      std::cout << "edge mismatch\n";
    if (iic != eec)
      std::cout << "edge mismatch c\n";
  }

  std::cout << std::distance(graph.begin(), graph.end()) << ":" << std::distance(graphc.begin(), graphc.end()) << "\n";
  std::cout << graph.size() << ":" << graphc.size() << "\n";

  std::cout << "BFS CSR\n";
  AsyncBFS<Graph>()(graph, sourcearg, "CSR");
  std::cout << "BFS CCSR\n";
  AsyncBFS<GraphC>()(graphc, sourcearg, "CCSR");
  std::cout << "Done BFS\n";

  auto size = graph.size();
  raw.resize(size);
  delta.resize(size);
  lenBW.resize(11);

  Galois::do_all(graph.begin(), graph.end(), ComputeRatio(), Galois::do_all_steal(true));

  if (dostat) {
    std::ofstream of(outfilename.c_str());
    of << "Type,Ind,Val\n";
    dumphist(of, "raw",raw);
    dumphist(of, "delta",delta);
  }

  std::cout << "Total Size (64bit): " << total_elem * 8 << "\n";
  std::cout << "Total Size (32bit): " << total_elem * 4 << "\n";
  std::cout << "Compressed Size (BW): " << total_bytesBW << "\n";
  std::cout << "Ratio (BW64bit): " << (double)total_bytesBW / ((double)total_elem * 8) << "\n";
  std::cout << "Ratio (BW32bit): " << (double)total_bytesBW / ((double)total_elem * 4) << "\n";

  dumphist(std::cout, "BW", lenBW);

  return 0;

  auto p = tryHuffDeltaOnly();
  auto hlen = (p.first + 7) / 8;
  std::cout << "Compressed Size (HuffDO): " << hlen << "\n";
  std::cout << "HuffDO Table Size: " << p.second << "\n";
  std::cout << "Ratio (HuffDO32bit): " << (double)hlen / ((double)total_elem * 4) << "\n";

  p = tryHuff();
  hlen = (p.first + 7) / 8;
  std::cout << "Compressed Size (Huff): " << hlen << "\n";
  std::cout << "Huff Table Size: " << p.second << "\n";
  std::cout << "Ratio (Huff64bit): " << (double)hlen / ((double)total_elem * 8) << "\n";
  std::cout << "Ratio (Huff32bit): " << (double)hlen / ((double)total_elem * 4) << "\n";

  return 0;
}
/*
library(ggplot2)                                      
wl <- read.csv(file="out.hist",sep=",", head=TRUE)
qplot(Ind, Val, data=wl, colour=Type, shape=Type, geom="point") + geom_line()
*/
