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

#include "galois/Timer.h"
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <fstream>

template<typename Iter>
Iter vByteEnc(uint32_t tmp, Iter out) {
  int numbits = 32 - __builtin_clz(tmp|1);
  if (numbits <= 7) {
    *out++ = tmp;
  } else if (numbits <= 14) {
    *out++ = (0x80 | (0x7F & (tmp >> 7)));
    *out++ = (       (0x7F & (tmp >> 0)));
  } else if (numbits <= 21) {
    *out++ = (0x80 | (0x7F & (tmp >> 14)));
    *out++ = (0x80 | (0x7F & (tmp >> 7)));
    *out++ = (       (0x7F & (tmp >> 0)));
  } else if (numbits <= 28) {
    *out++ = (0x80 | (0x7F & (tmp >> 21)));
    *out++ = (0x80 | (0x7F & (tmp >> 14)));
    *out++ = (0x80 | (0x7F & (tmp >> 7)));
    *out++ = (       (0x7F & (tmp >> 0)));
  } else {
    *out++ = (0x80 | (0x7F & (tmp >> 28)));
    *out++ = (0x80 | (0x7F & (tmp >> 21)));
    *out++ = (0x80 | (0x7F & (tmp >> 14)));
    *out++ = (0x80 | (0x7F & (tmp >> 7)));
    *out++ = (       (0x7F & (tmp >> 0)));
  }
  return out;
}

template<typename IterIn, typename IterOut>
void vByteEncode(IterIn b, IterIn e, IterOut o) {
  while (b != e)
    o = vByteEnc(*b++, o);
}

uint32_t vByteDec(uint8_t* in) {
  uint8_t rd = *in++;
  uint32_t retval = rd & 0x7F;
  if (rd & 0x80) {
    rd = *in++;
    retval <<= 7;
    retval |= rd & 0x7F;
    if (rd & 0x80) {
      rd = *in++;
      retval <<= 7;
      retval |= rd & 0x7F;
      if (rd & 0x80) {
        rd = *in++;
        retval <<= 7;
        retval |= rd & 0x7F;
        if (rd & 0x80) {
          rd = *in;
          retval <<= 7;
          retval |= rd & 0x7F;
        }
      }
    }
  }
  return retval;
}

uint32_t vByteDecSkip(uint8_t*& in) {
  uint8_t rd = *in++;
  uint32_t retval = rd & 0x7F;
  if (rd & 0x80) {
    rd = *in++;
    retval <<= 7;
    retval |= rd & 0x7F;
    if (rd & 0x80) {
      rd = *in++;
      retval <<= 7;
      retval |= rd & 0x7F;
      if (rd & 0x80) {
        rd = *in++;
        retval <<= 7;
        retval |= rd & 0x7F;
        if (rd & 0x80) {
          rd = *in++;
          retval <<= 7;
          retval |= rd & 0x7F;
        }
      }
    }
  }
  return retval;
}

uint8_t* vByteSkip(uint8_t* in) {
  uint8_t rd = *in++;
  if (rd & 0x80) {
    rd = *in++;
    if (rd & 0x80) {
      rd = *in++;
      if (rd & 0x80) {
        rd = *in++;
        if (rd & 0x80) {
          in++;
        }
      }
    }
  }
  return in;
}

template<typename Iter>
Iter v2ByteEnc(uint32_t tmp, Iter out) {
  int numbits = 32 - __builtin_clz(tmp|1);
  tmp <<= 2;
  if (numbits <= 6) {
    *out++ = 0xFF & tmp;
  } else if (numbits <= 14) {
    *out++ = 0xFF & (tmp | 0x01);
    *out++ = 0xFF & (tmp >> 8);
  } else if (numbits <= 22) {
    *out++ = 0xFF & (tmp | 0x02);
    *out++ = 0xFF & (tmp >>  8);
    *out++ = 0xFF & (tmp >> 16);
  } else if (numbits <= 30) {
    *out++ = 0xFF & (tmp | 0x03);
    *out++ = 0xFF & (tmp >>  8);
    *out++ = 0xFF & (tmp >> 16);
    *out++ = 0xFF & (tmp >> 24);
  } else {
    abort();
  }
  return out;
}

template<typename IterIn, typename IterOut>
void v2ByteEncode(IterIn b, IterIn e, IterOut o) {
  while (b != e)
    o = v2ByteEnc(*b++, o);
}

uint32_t v2ByteDec(uint8_t* in) {
  uint32_t rd = *reinterpret_cast<uint32_t*>(in);
  auto num = rd & 0x03;
  static uint32_t shiftTbl[4] = {0x000000FF, 0x0000FFFF, 0x00FFFFFF, 0xFFFFFFFF};
  return (rd & shiftTbl[num]) >> 2;
}

uint8_t* v2ByteSkip(uint8_t* in) {
  uint8_t rd = *in;
  auto num = rd & 0x03;
  return in + num + 1;
}

uint32_t v2ByteDecSkip(uint8_t*& in) {
  uint32_t rd = *reinterpret_cast<uint32_t*>(in);
  auto num = rd & 0x03;
  in += num + 1;
  static uint32_t shiftTbl[4] = {0x000000FF, 0x0000FFFF, 0x00FFFFFF, 0xFFFFFFFF};
  return (rd & shiftTbl[num]) >> 2;
}

class vByteIterator : public std::iterator<std::forward_iterator_tag, uint32_t> {
  uint8_t* base;

public:
  vByteIterator(uint8_t* b = nullptr) :base(b) {}
  
  uint32_t operator*() {
    return vByteDec(base);
  }

  vByteIterator& operator++() {
    base = vByteSkip(base);
    return *this;
  }

  vByteIterator operator++(int) {
    vByteIterator retval(*this);
    ++(*this);
    return retval;
  }

  bool operator==(const vByteIterator& rhs) const { return base == rhs.base; }
  bool operator!=(const vByteIterator& rhs) const { return base != rhs.base; }
};

class v2ByteIterator : public std::iterator<std::forward_iterator_tag, uint32_t> {
  uint8_t* base;

public:
  v2ByteIterator(uint8_t* b = nullptr) :base(b) {}
  
  uint32_t operator*() {
    return v2ByteDec(base);
  }

  v2ByteIterator& operator++() {
    base = v2ByteSkip(base);
    return *this;
  }

  v2ByteIterator operator++(int) {
    v2ByteIterator retval(*this);
    ++(*this);
    return retval;
  }

  bool operator==(const v2ByteIterator& rhs) const { return base == rhs.base; }
  bool operator!=(const v2ByteIterator& rhs) const { return base != rhs.base; }
};

class v2DeltaIterator : public std::iterator<std::forward_iterator_tag, uint32_t> {
  uint8_t* base;
  uint32_t state;

public:
  v2DeltaIterator(uint8_t* b = nullptr) :base(b), state(0) {}
  
  uint32_t operator*() {
    return state + v2ByteDec(base);
  }

  v2DeltaIterator& operator++() {
    state += v2ByteDecSkip(base);
    //base = v2ByteSkip(base);
    return *this;
  }

  v2DeltaIterator operator++(int) {
    v2DeltaIterator retval(*this);
    ++(*this);
    return retval;
  }

  bool operator==(const v2DeltaIterator& rhs) const { return base == rhs.base; }
  bool operator!=(const v2DeltaIterator& rhs) const { return base != rhs.base; }
};

class vDeltaIterator : public std::iterator<std::forward_iterator_tag, uint32_t> {
  uint8_t* base;
  uint32_t state;

public:
  vDeltaIterator(uint8_t* b = nullptr) :base(b), state(0) {}
  
  uint32_t operator*() {
    return state + vByteDec(base);
  }

  vDeltaIterator& operator++() {
    state += vByteDecSkip(base);
    //base = vByteSkip(base);
    return *this;
  }

  vDeltaIterator operator++(int) {
    vDeltaIterator retval(*this);
    ++(*this);
    return retval;
  }

  bool operator==(const vDeltaIterator& rhs) const { return base == rhs.base; }
  bool operator!=(const vDeltaIterator& rhs) const { return base != rhs.base; }
};

class deltaIterator : public std::iterator<std::forward_iterator_tag, uint32_t> {
  uint32_t* base;
  uint32_t state;

public:
  deltaIterator(uint32_t* b = nullptr) :base(b), state(0) {}
  
  uint32_t operator*() {
    return state + *base;
  }

  deltaIterator& operator++() {
    state += *base;
    ++base;
    return *this;
  }

  deltaIterator operator++(int) {
    deltaIterator retval(*this);
    ++(*this);
    return retval;
  }

  bool operator==(const deltaIterator& rhs) const { return base == rhs.base; }
  bool operator!=(const deltaIterator& rhs) const { return base != rhs.base; }
};


template<typename Iter, typename Iter2>
void deltaCode(Iter b, Iter e, Iter2 out) {
  auto val = 0;
  while (b != e) {
    auto tmp = *b++;
    assert (tmp >= val);
    *out++ = tmp - val;
    val = tmp;
  }
}

template<typename Iter, typename Iter2>
void rleCode(Iter b, Iter e, Iter2 out) {
  while (b != e) {
    auto val = *b++;
    *out++ = val;
    if (val <= 1) {
      decltype(val) run = 0;
      while (b != e && *b == val) {
        ++run;
        ++b;
      }
      *out++ = run;
    }
  }
}

namespace galois {
namespace graphs {
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
  typedef v2DeltaIterator edge_iterator;
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
    galois::runtime::acquire(&nodeData[N], mflag);
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
  node_data_reference getData(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    // galois::runtime::checkWrite(mflag, false);
    NodeInfo& NI = nodeData[N];
    acquireNode(N, mflag);
    return NI.getData();
  }

  edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::UNPROTECTED) {
    // galois::runtime::checkWrite(mflag, false);
    return edgeData[*ni];
  }

  GraphNode getEdgeDst(edge_iterator ni) {
    return *ni;
  }

  uint64_t size() const { return numNodes; }
  uint64_t sizeEdges() const { return numEdges; }

  iterator begin() const { return iterator(0); }
  iterator end() const { return iterator(numNodes); }

  const_local_iterator local_begin() const { return const_local_iterator(this->localBegin(numNodes)); }
  const_local_iterator local_end() const { return const_local_iterator(this->localEnd(numNodes)); }
  local_iterator local_begin() { return local_iterator(this->localBegin(numNodes)); }
  local_iterator local_end() { return local_iterator(this->localEnd(numNodes)); }

  edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    if (galois::runtime::shouldLock(mflag)) {
      for (edge_iterator ii = raw_begin(N), ee = raw_end(N); ii != ee; ++ii) {
        acquireNode(*ii, mflag);
      }
    }
    return raw_begin(N);
  }

  edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    acquireNode(N, mflag);
    return raw_end(N);
  }

  detail::EdgesIterator<LC_CCSR_Graph> out_edges(GraphNode N, MethodFlag mflag = MethodFlag::WRITE) {
    return detail::EdgesIterator<LC_CCSR_Graph>(*this, N, mflag);
  }

  void allocateFrom(FileGraph& graph) {
    numNodes = graph.size();
    numEdges = graph.sizeEdges();
    if (UseNumaAlloc) {
      nodeData.allocateLocal(numNodes);
      edgeIndData.allocateLocal(numNodes);
      edgeDst.allocateLocal(numEdges*sizeof(uint32_t));
      edgeData.allocateLocal(numEdges);
      this->outOfLineAllocateLocal(numNodes);
    } else {
      nodeData.allocateInterleaved(numNodes);
      edgeIndData.allocateInterleaved(numNodes);
      edgeDst.allocateInterleaved(numEdges * sizeof(uint32_t));
      edgeData.allocateInterleaved(numEdges);
      this->outOfLineAllocateInterleaved(numNodes);
    }
  }

  void constructFrom(FileGraph& graph, unsigned tid, unsigned total) {
    auto r = graph.divideByNode(
        NodeData::size_of::value + EdgeIndData::size_of::value + LC_CCSR_Graph::size_of_out_of_line::value,
        EdgeDst::size_of::value + EdgeData::size_of::value,
        tid, total).first;
    this->setLocalRange(*r.first, *r.second);
    if (tid == 0) {
      uint64_t offset = 0;
      for (FileGraph::iterator ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
        nodeData.constructAt(*ii);
        this->outOfLineConstructAt(*ii);
        std::vector<uint32_t> dsts, dsts2, dsts3;
        for (FileGraph::edge_iterator nn = graph.edge_begin(*ii), en = graph.edge_end(*ii); nn != en; ++nn) {
          if (EdgeData::has_value)
            edgeData.set(*nn, graph.getEdgeData<typename EdgeData::value_type>(nn));
          dsts.push_back(graph.getEdgeDst(nn));
        }
        std::sort(dsts.begin(), dsts.end());
        deltaCode(dsts.begin(), dsts.end(), std::back_inserter(dsts2));
        //rleCode(dsts2.begin(), dsts2.end(), std::back_inserter(dsts3));
        std::vector<uint8_t> bw;
        v2ByteEncode(dsts2.begin(), dsts2.end(), std::back_inserter(bw));
        std::copy(bw.begin(), bw.end(), &edgeDst[offset]);
        offset += bw.size();
        // std::copy(dsts2.begin(), dsts2.end(), &edgeDst[offset]);
        // offset += dsts2.size();
        edgeIndData[*ii] = offset;
        // auto foo = decodeOne(&*bw.begin(), 0);
        // std::cout << "\n" << *dsts.begin() << " " << foo.first << " " << (foo.second - &*bw.begin()) << "\n\n";
      }
      edgeDst[offset] = 0x00;
      std::cout << "Final Offset " << offset << "\n";
    }
  }
};

} // end namespace
} // end namespace


typedef galois::graphs::LC_CSR_Graph<unsigned, void> Graph;
typedef galois::graphs::LC_CCSR_Graph<unsigned, void> GraphC;

namespace cll = llvm::cl;
static cll::opt<std::string> filename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outfilename(cll::Positional, cll::desc("<output file>"), cll::Optional);
static cll::opt<unsigned> sourcearg("source", cll::init(0), cll::desc("source for bfs"), cll::Optional);
bool dostat;

std::vector<unsigned int> raw, delta;
std::vector<unsigned int> lenBW;

unsigned long int total_elem = 0;
unsigned long int total_bytesBW = 0, total_bytesBW2 = 0;

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

    void operator()(WorkItem& item, galois::UserContext<WorkItem>& ctx) const {
      GNode n = item.first;

      unsigned newDist = item.second;
      if (newDist > gr.getData(n, galois::MethodFlag::UNPROTECTED))
        return;

      ++newDist;

      for (typename Gr::edge_iterator ii = gr.edge_begin(n, galois::MethodFlag::UNPROTECTED),
             ei = gr.edge_end(n, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = gr.getEdgeDst(ii);
        volatile unsigned* ddata = &gr.getData(dst, galois::MethodFlag::UNPROTECTED);
        
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
    using namespace galois::worklists;
    typedef dChunkedFIFO<64> dChunk;
    //typedef ChunkedFIFO<64> Chunk;
    typedef OrderedByIntegerMetric<Indexer,dChunk> OBIM;
    
    galois::do_all(graph, [&graph] (const GNode& n) { graph.getData(n) = ~0; }, galois::loopname("init"));

    graph.getData(source) = 0;

    galois::for_each(WorkItem(source, 0), Process(graph), galois::wl<OBIM>(), galois::loopname(name));
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
  void operator()(const GNode& n) const {
    std::deque<unsigned int> IDs, IDs2, IDs3;
    std::vector<uint8_t> var;
    for (Graph::edge_iterator ii = graph.edge_begin(n),
           ee = graph.edge_end(n); ii != ee; ++ii)
      IDs.push_back(graph.getEdgeDst(ii));
    std::sort(IDs.begin(), IDs.end());
    if (dostat)
      hist(IDs.begin(), IDs.end(), raw);
    deltaCode(IDs.begin(), IDs.end(), std::back_inserter(IDs2));
    if (dostat)
      hist(IDs.begin(), IDs.end(), delta);
    rleCode(IDs2.begin(), IDs2.end(), std::back_inserter(IDs3));
    __sync_fetch_and_add(&total_elem, IDs.size());
    vByteEncode(IDs3.begin(), IDs3.end(), std::back_inserter(var));
    __sync_fetch_and_add(&total_bytesBW, var.size());
    var.clear();
    v2ByteEncode(IDs3.begin(), IDs3.end(), std::back_inserter(var));
    __sync_fetch_and_add(&total_bytesBW2, var.size());

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
    symbols.push_back(HNode {n1,n2,0,n1->freq + n2->freq});
    Q.push(&symbols.back());
  }
  return Q.top();
}

void buildTableInternal(codeEntry e, HNode* n, std::deque<codeEntry>& codes) {
  if (n->left) {
    buildTableInternal(codeEntry {0, e.nbits + 1, e.code << 1}, n->left, codes);
    buildTableInternal(codeEntry {0, e.nbits + 1, 1 | (e.code << 1)}, n->right, codes);
  } else {
    e.val = n->val;
    codes.push_back(e);
  }
}

std::deque<codeEntry> buildTable(HNode* root) {
  std::deque<codeEntry> retval;
  buildTableInternal(codeEntry {0,0,0}, root, retval);
  return retval;
}

std::pair<unsigned long, unsigned> tryHuff() {
  std::map<uint32_t, HNode*> hyst;
  std::deque<HNode> symbols;
  std::deque<uint32_t> data;

  for (auto ni = graph.begin(), ne = graph.end(); ni != ne; ++ni) {
    std::deque<uint32_t> local, local2;;
    for (auto ei = graph.edge_begin(*ni), ee = graph.edge_end(*ni); ei != ee; ++ei) {
      local.push_back(graph.getEdgeDst(ei));
    }
    std::sort(local.begin(), local.end());
    deltaCode(local.begin(), local.end(), std::back_inserter(local2));
    std::copy(local2.begin(), local2.end(), std::back_inserter(data));
    for (auto ii = local.begin(), ee = local.end(); ii != ee; ++ii) {
      HNode*& n = hyst[*ii];
      if (!n) {
        symbols.push_back(HNode {nullptr, nullptr, *ii, 0});
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
    std::deque<uint32_t> local, local2;
    for (auto ei = graph.edge_begin(*ni), ee = graph.edge_end(*ni); ei != ee; ++ei) {
      local.push_back(graph.getEdgeDst(ei));
    }
    std::sort(local.begin(), local.end());
    deltaCode(local.begin(), local.end(), std::back_inserter(local2));
    std::copy(local2.begin() + 1, local2.end(), std::back_inserter(data));
    total += 4 * 8;
    for (auto ii = local.begin() + 1, ee = local.end(); ii != ee; ++ii) {
      HNode*& n = hyst[*ii];
      if (!n) {
        symbols.push_back(HNode {nullptr, nullptr, *ii, 0});
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
  galois::StatManager statManager;
  LonestarStart(argc, argv, 0,0,0);

  if (false) {
    std::cout << std::hex;
    std::ostream_iterator<int> out_it (std::cout,", ");

    for (uint32_t x = 0; x < ~0U; ++x) {
      std::vector<uint32_t> src;
      std::vector<uint8_t> dv1, dv2;
      src.push_back(x);
      vByteEncode(src.begin(), src.end(), std::back_inserter(dv1));
      v2ByteEncode(src.begin(), src.end(), std::back_inserter(dv2));
 
      if (vByteDec(&dv1[0]) != v2ByteDec(&dv2[0])) {
        std::copy ( src.begin(), src.end(), out_it );
        std::cout << "\t";
        std::copy ( dv1.begin(), dv1.end(), out_it );
        std::cout << "\t";
        std::copy ( dv2.begin(), dv2.end(), out_it );
        std::cout << "\t" << vByteDec(&dv1[0]) << "\t" << v2ByteDec(&dv2[0]) << "\n";
        return 1;
      }
    }

    std::vector<uint32_t> src[3];
    std::vector<uint8_t> dsts[6];

    for (int i = 0; i < 32; ++i)
      src[0].push_back(i);
    for (int i = 250; i < 270; ++i)
      src[0].push_back(i);
    for (int i = 16777206; i < 16777226; ++i)
      src[0].push_back(i);
    
    deltaCode(src[0].begin(), src[0].end(), std::back_inserter(src[1]));
    rleCode(src[1].begin(), src[1].end(), std::back_inserter(src[2]));

    for (int x = 0; x < 3; ++x) {
      std::cout << x << ":" << std::distance(src[x].begin(), src[x].end()) << " ";
      std::copy ( src[x].begin(), src[x].end(), out_it );
      std::cout << "\n";
    }
    std::cout << "\n";

    vByteEncode(src[0].begin(), src[0].end(), std::back_inserter(dsts[0]));
    v2ByteEncode(src[0].begin(), src[0].end(), std::back_inserter(dsts[1]));
    vByteEncode(src[1].begin(), src[1].end(), std::back_inserter(dsts[2]));
    v2ByteEncode(src[1].begin(), src[1].end(), std::back_inserter(dsts[3]));
    vByteEncode(src[2].begin(), src[2].end(), std::back_inserter(dsts[4]));
    v2ByteEncode(src[2].begin(), src[2].end(), std::back_inserter(dsts[5]));

    for (int x = 0; x < 6; ++x) {
      std::cout << x << ":" << std::distance(dsts[x].begin(), dsts[x].end()) << " ";
      std::copy ( dsts[x].begin(), dsts[x].end(), out_it );
      std::cout << "\n";
    }
    std::cout << "\n";

    std::copy ( vByteIterator(&dsts[0][0]), vByteIterator(&dsts[0][dsts[0].size()]), out_it );
    std::cout << "\n";
    std::copy ( v2ByteIterator(&dsts[1][0]), v2ByteIterator(&dsts[1][dsts[1].size()]), out_it );
    std::cout << "\n";
    std::copy ( vByteIterator(&dsts[2][0]), vByteIterator(&dsts[2][dsts[2].size()]), out_it );
    std::cout << "\n";
    std::copy ( v2ByteIterator(&dsts[3][0]), v2ByteIterator(&dsts[3][dsts[3].size()]), out_it );
    std::cout << "\n";
    std::copy ( vByteIterator(&dsts[4][0]), vByteIterator(&dsts[4][dsts[4].size()]), out_it );
    std::cout << "\n";
    std::copy ( v2ByteIterator(&dsts[5][0]), v2ByteIterator(&dsts[5][dsts[5].size()]), out_it );
    std::cout << "\n";

    // rleCode(dsts2.begin(), dsts2.end(), std::back_inserter(dsts3));
    // std::vector<uint8_t> bw = varCodeBW(dsts3.begin(), dsts3.end());
    // bw.push_back(0x00);
    // for (decompIterator ii(&bw.front()), ee(&bw.back()); ii != ee; ++ii)
    //   std::cout << *ii << " ";
    // std::cout << "\n";
    
    std::cout << std::dec;
  }

  dostat = outfilename.size() > 1;
  if (dostat)
    std::cout << "Collecting All Histograms\n";

  galois::graphs::readGraph(graph, filename);
  //galois::graphs::readGraph(graphc, filename);

  // for (unsigned int x = 0; x < 0; ++x) {
  //   auto ii = graph.edge_begin(x);
  //   auto iic = graphc.edge_begin(x);
  //   auto ee = graph.edge_end(x);
  //   auto eec = graphc.edge_end(x);
  //   int count = 0;
  //   while (ii != ee && iic != eec) {
  //     if (graph.getEdgeDst(ii) != graphc.getEdgeDst(iic)) {
  //       std::cout << "Mismatch at " << x << "," << count << " : " << graph.getEdgeDst(ii) << " " <<  graphc.getEdgeDst(iic) << "\n";
  //     }
  //     ++count;
  //     ++ii;
  //     ++iic;
  //   }
  //   if (ii != ee) 
  //     std::cout << "edge mismatch\n";
  //   if (iic != eec)
  //     std::cout << "edge mismatch c\n";
  // }

  // std::cout << std::distance(graph.begin(), graph.end()) << ":" << std::distance(graphc.begin(), graphc.end()) << "\n";
  // std::cout << graph.size() << ":" << graphc.size() << "\n";

  std::cout << "BFS CSR " << sourcearg << "\n";
  AsyncBFS<Graph>()(graph, sourcearg, "CSR");
  // std::cout << "BFS CSSR " << sourcearg << "\n";
  // AsyncBFS<GraphC>()(graphc, sourcearg, "CCSR");
  std::cout << "Done BFS " << sourcearg << "\n";

  return 0;

  auto size = graph.size();
  raw.resize(size);
  delta.resize(size);
  lenBW.resize(11);

  galois::do_all(graph.begin(), graph.end(), ComputeRatio(), galois::steal());

  if (dostat) {
    std::cout << "Writing to " << outfilename.c_str() << "\n";
    std::ofstream of(outfilename.c_str());
    of << "Type,Ind,Val\n";
    dumphist(of, "raw",raw);
    dumphist(of, "delta",delta);
  }

  std::cout << "Total Size (64bit): " << total_elem * 8 << "\n";
  std::cout << "Total Size (32bit): " << total_elem * 4 << "\n";
  std::cout << "Compressed Size (BW): " << total_bytesBW << "\n";
  std::cout << "Compressed Size (BW2): " << total_bytesBW2 << "\n";
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
