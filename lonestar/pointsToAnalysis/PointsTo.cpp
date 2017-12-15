/** Points-to Analysis application -*- C++ -*-
 * @file
 *
 * An inclusion-based points-to analysis algorithm to demostrate the Galois 
 * system.
 *
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 * @author Rupesh Nasre <rupesh0508@gmail.com>
 * @author Loc Hoang <l_hoang@utexas.edu> (mainly fixes/documentation/cleanup)
 */
#include "galois/Galois.h"
#include "galois/graphs/Graph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include <fstream>
#include <deque>
#include "SparseBitVector.h"

////////////////////////////////////////////////////////////////////////////////
// Command line parameters
////////////////////////////////////////////////////////////////////////////////

namespace cll = llvm::cl;

const char* name = "Points-to Analysis";
const char* desc = "Performs inclusion-based points-to analysis over the input "
                   "constraints.";
const char* url = NULL;

static cll::opt<std::string> input(cll::Positional, 
                                   cll::desc("Constraints file"), 
                                   cll::Required);

static cll::opt<bool> useSerial("serial",
                                cll::desc("Runs serial version of the algorithm "
                                          "(i.e. 1 thread, no galois::for_each)"), 
                                cll::init(false));

const unsigned THRESHOLD_LS = 500000;
const unsigned THRESHOLD_OCD = 500;

////////////////////////////////////////////////////////////////////////////////
// Declaration of strutures, types, and variables
////////////////////////////////////////////////////////////////////////////////

/**
 * Structure of a node in the graph.
 */
struct Node {
  unsigned id;

  Node() {
    id = 0;
  }

  Node(unsigned lid) : id(lid) { }
};

using Graph = galois::graphs::FirstGraph<Node, void, true>;
using GNode = Graph::GraphNode;
using UpdatesVec = galois::gstl::Vector<GNode>;

/**
 * Class representing a points-to constraint.
 */
class PtsToCons {
 public:
  using ConstraintType = enum {AddressOf = 0, Copy, Load, Store};
 private:
  unsigned src, dst;
  ConstraintType type;
 public:
  PtsToCons(ConstraintType tt, unsigned ss, unsigned dd) {
    src = ss;
    dst = dd;
    type = tt;
  }

  /**
   * @returns This constraint's src and dst node
   */
  std::pair<unsigned, unsigned> getSrcDst() const {
    return std::pair<unsigned, unsigned>(src, dst);
  }

  /**
   * @returns The type of this constraint
   */
  ConstraintType getType() const {
    return type;
  }

  /**
   * Print out this constraint to stderr
   */
  void print() const {
    if (type == Store) {
      std::cerr << "*";
    }

    std::cerr << "v" << dst;
    std::cerr << " = ";

    if (type == Load) {
      std::cerr << "*";
    } else if (type == AddressOf) {
      std::cerr << "&";
    }

    std::cerr << "v" << src;
    std::cerr << std::endl;
  }
};

class PTA {
  using PointsToConstraints = galois::gstl::Vector<PtsToCons>;
  using PointsToInfo = galois::gstl::Vector<galois::SparseBitVector>;

 private:
  Graph graph;
  galois::gstl::Vector<GNode> nodes;
  PointsToInfo result; // pointsTo results for nodes

  PointsToConstraints addressCopyConstraints; 
  PointsToConstraints loadStoreConstraints;
  size_t numNodes = 0;

  /** 
   * Online Cycle Detection and elimination structure + functions.
   */
  struct OnlineCycleDetection { 
   private:
    PTA& outer;
    galois::gstl::Vector<unsigned> ancestors;
    galois::gstl::Vector<bool> visited;
    galois::gstl::Vector<unsigned> representative;
    galois::gstl::Vector<GNode> news; // TODO find better way to do this
    unsigned NoRepresentative; // "constant" that represents no representative

    /**
     * @returns true of the nodeid is an ancestor node
     */
    bool isAncestor(unsigned nodeid) {
      for (unsigned ii : ancestors) {
        if (ii == nodeid) {
          return true;
        }
      }
      return false;
    }

    /**
     * Depth first recursion of the nodeid to see if it eventually
     * reaches an ancestor, in which case there is a cycle. The cycle is
     * then collapsed, i.e. all nodes in the cycle have their representative
     * changed to the representative of the node where the cycle starts.
     *
     * Note it is okay not to detect all cycles as it is only an efficiency 
     * concern.
     *
     * @param nodeid nodeid to check the existance of a cycle
     * @param cyclenode This is used as OUTPUT parameter; if a node is
     * detected to be an ancestor, it is returned via this variable.
     * @returns true if a cycle has been detected (i.e. a node has a path
     * to an ancestor), false otherwise
     */
    bool cycleDetect(unsigned nodeid, unsigned &cyclenode) {  
      unsigned nodeRep = getFinalRepresentative(nodeid);

      // if the node is an ancestor, that means there's a path from the ancestor
      // to the ancestor (i.e. cycle)
      if (isAncestor(nodeRep)) {
        cyclenode = nodeRep;
        return true;
      }

      if (visited[nodeRep]) {
        return false;
      }

      visited[nodeRep] = true;

      // keep track of the current depth first search path
      ancestors.push_back(nodeRep);

      GNode nn = outer.nodes[nodeRep];

      for (auto ii : outer.graph.edges(nn, galois::MethodFlag::UNPROTECTED)) {
        // get id of destination node
        Node& nn = outer.graph.getData(outer.graph.getEdgeDst(ii), 
                                       galois::MethodFlag::UNPROTECTED);
        unsigned destID = nn.id;

        // recursive depth first cycle detection; if a cycle is found,
        // collapse the path
        if (cycleDetect(destID, cyclenode)) {
          cycleCollapse(cyclenode);
        }
      }
      ancestors.pop_back();

      return false;
    }

    /**
     * Make all nodes that are a part of some detected cycle starting at
     * repr have their representatives changed to the representative of
     * repr (thereby "collapsing" the cycle).
     *
     * @param repr The node at which the cycle begins
     */
    void cycleCollapse(unsigned repr) {
      // assert(repr is present in ancestors).
      unsigned repToChangeTo = getFinalRepresentative(repr);

      for (auto ii = ancestors.begin(); ii != ancestors.end(); ++ii) {
        if (*ii == repr) {
          galois::gDebug("collapsing cycle for ", repr);
          // cycle exists between nodes ancestors[*ii..end].
          for (auto jj = ii; jj != ancestors.end(); ++jj) {
            unsigned jjrepr = getFinalRepresentative(*jj);  // jjrepr has no representative.
            makeRepr(jjrepr, repToChangeTo);
          }
          break;
        }
      }

      news.push_back(outer.nodes[repr]);
      news.push_back(outer.nodes[repToChangeTo]);
    }

    /**
     * Make repr the representative of nodeid.
     *
     * @param nodeid node to change the representative of
     * @param repr nodeid will have its representative changed to this
     */
    void makeRepr(unsigned nodeid, unsigned repr) {
      if (repr != nodeid) {
        galois::gDebug("repr[", nodeid, "] = ", repr);
        representative[nodeid] = repr;

        // the representative needs to have all of the items that the nodes
        // it is representing has, so if the node has more than the rep,
        // unify
        if (!outer.result[nodeid].isSubsetEq(outer.result[repr])) {
          //outer.graph.getData(nodes[repr]); // lock it TODO figure if nec.
          outer.result[repr].unify(outer.result[nodeid]);
        }

        // push regardless because the node in question might have
        // been "updated" by virtue of the representative being updated
        // elsewhere in this cycle detection process
        news.push_back(outer.nodes[nodeid]); 
      }
    }

   public:

    OnlineCycleDetection(PTA& o) : outer(o) {}

    /**
     * Init fields (outer needs to have numNodes set).
     */
    void init() {
      NoRepresentative = outer.numNodes;
      visited.resize(outer.numNodes);
      representative.resize(outer.numNodes);

      for (unsigned ii = 0; ii < outer.numNodes; ++ii) {
        representative[ii] = NoRepresentative;
      }
    }

    /**
     * Given a node id, find its representative. Also, do path compression
     * of the path to the representative.
     *
     * @param nodeid Node id to get the representative of
     * @returns The representative of nodeid
     */
    unsigned getFinalRepresentative(unsigned nodeid) {
      unsigned finalRep = nodeid;

      // Follow chain of representatives until a "root" is reached
      while (representative[finalRep] != NoRepresentative) {
        finalRep = representative[finalRep];
      }

      // TODO figure out if path compression might screw things up
      // path compression; make all things along path to final representative
      // point to the final representative
      unsigned curRep = representative[nodeid];

      while (curRep != NoRepresentative) {
        representative[nodeid] = finalRep;
        nodeid = curRep;
        curRep = representative[nodeid];
      }

      return finalRep;
    }

    /**
     * Go over all sources of new edges to see if there are cycles in them.
     * If so, collapse the cycles.
     *
     * @param updates vector of nodes that are sources of new edges.
     */
    template<typename VecType>
    void process(VecType& updates) { 
      galois::gPrint(getFinalRepresentative(33659), "\n");
      //return; // TODO this function (or something it calls) is not completely 
                // correct FIXME

      // TODO this can probably be made more efficient (fill?)
      for (unsigned ii = 0; ii < outer.numNodes; ++ii) {
        visited[ii] = false;
      }

      // TODO don't use news to add to the worklist; find a more efficient way
      news.clear();

      unsigned cyclenode = NoRepresentative;  // set to invalid id.

      for (const GNode& up : updates) {
        Node &nn = outer.graph.getData(up, galois::MethodFlag::UNPROTECTED);
        unsigned nodeid = nn.id;
        galois::gDebug("cycle process ", nodeid);

        if (cycleDetect(nodeid, cyclenode)) {
          cycleCollapse(cyclenode);
        }
      }

      // TODO find a more efficient way to do this
      for (auto d : news) {
        updates.push_back(d);
      }
    }
  }; // end struct OnlineCycleDetection

  OnlineCycleDetection ocd; // cycle detector/squasher

  // TODO better documentation
  // add edges to the graph based on points-to information of the nodes
  // and then add the source of each edge to the updates.
  template <typename C>
  void processLoadStore(const PointsToConstraints &constraints, C& ctx, 
                        galois::MethodFlag flag = galois::MethodFlag::WRITE) {
    for (auto& constraint : constraints) {
      //std::cout << "debug: Processing constraint: "; constraint->print();
      unsigned src, dst;
      std::tie(src, dst) = constraint.getSrcDst();

      unsigned srcRepr = ocd.getFinalRepresentative(src);
      unsigned dstRepr = ocd.getFinalRepresentative(dst);

      if (constraint.getType() == PtsToCons::Load) { 
        galois::gstl::Vector<unsigned> ptsToOfSrc;
        result[srcRepr].getAllSetBits(ptsToOfSrc);

        GNode &nDstRepr = nodes[dstRepr]; // id of dst in graph

        for (unsigned pointee : ptsToOfSrc) {
          unsigned pointeeRepr = ocd.getFinalRepresentative(pointee);

          // add edge from pointee to dst if it doesn't already exist
          if (pointeeRepr != dstRepr && 
              graph.findEdge(nodes[pointeeRepr], nodes[dstRepr]) == 
                  graph.edge_end(nodes[pointeeRepr])) {
            GNode &nPointeeRepr = nodes[pointeeRepr];

            galois::gDebug("adding edge from ", pointee, " to ", dst);

            graph.addEdge(nPointeeRepr, nDstRepr, flag);

            ctx.push_back(nPointeeRepr);
          }
        }
      } else {  // store whatever src has into whatever dst points to
        galois::gstl::Vector<unsigned> ptsToOfDst;
        result[dstRepr].getAllSetBits(ptsToOfDst);
        GNode& nSrcRepr = nodes[srcRepr];

        bool newEdgeAdded = false;

        for (unsigned pointee : ptsToOfDst) {
          unsigned pointeeRepr = ocd.getFinalRepresentative(pointee);

          // add edge from src -> pointee if it doesn't exist
          if (srcRepr != pointeeRepr && 
                graph.findEdge(nodes[srcRepr], nodes[pointeeRepr]) == 
                    graph.edge_end(nodes[srcRepr])) {

            galois::gDebug("adding edge from ", src, " to ", pointee);
            graph.addEdge(nSrcRepr, nodes[pointeeRepr], flag);
            newEdgeAdded = true;
          }
        }

        if (newEdgeAdded) {
          ctx.push_back(nSrcRepr);
        }
      }
    }
  }

  /**
   * Processes the AddressOf and Copy constraints. 
   *
   * Sets the bitvector for AddressOf constraints, i.e. a set bit means
   * that you point to whatever that bit represents.
   *
   * Creates edges for Copy constraints, i.e. edge from a to b indicates
   * b is a copy of a.
   *
   * @param constraints vector of AddressOf and Copy constraints
   * @returns vector of UpdatesRequests from all sources with new edges
   * added by the Copy constraint
   */
  template<typename VecType>
  VecType processAddressOfCopy(const PointsToConstraints& constraints) {
    VecType updates;

    for (auto ii = constraints.begin(); ii != constraints.end(); ++ii) {
      //std::cout << "debug: Processing constraint: "; ii->print();
      unsigned src;
      unsigned dst;
      std::tie(src, dst) = ii->getSrcDst();

      if (ii->getType() == PtsToCons::AddressOf) {
        if (result[dst].set(src)) { // this alters the state of the bitvector
          galois::gDebug("saving v", dst, "->v", src);
        }
      } else if (src != dst) {  // copy.
        GNode &nn = nodes[src];
        graph.addEdge(nn, nodes[dst], galois::MethodFlag::UNPROTECTED);
        galois::gDebug("adding edge from ", src, " to ", dst);

        updates.push_back(nn);
      }
    }

    return updates;
  }

  /**
   * If an edge exists from src to dst, then dst is a copy of src.
   * Propogate any points to information from source to dest.
   *
   * @param src Source node in graph
   * @param dst Dest node in graph
   * @param flag Flag specifying access to the graph. It's only used to 
   * acquire a lock on the destination's bitvector if the flag specifies
   * that a lock is necessary (e.g. WRITE)
   * @returns non-negative value if any bitvector has changed
   */
  unsigned propagate(GNode src, GNode dst, 
                     galois::MethodFlag flag = galois::MethodFlag::WRITE) {
    unsigned srcID = graph.getData(src, galois::MethodFlag::UNPROTECTED).id;
    unsigned dstID = graph.getData(dst, galois::MethodFlag::UNPROTECTED).id;

    unsigned newPtsTo = 0;

    if (srcID != dstID) {
      unsigned srcReprID = ocd.getFinalRepresentative(srcID);
      unsigned dstReprID = ocd.getFinalRepresentative(dstID);

      // if src is a not subset of dst... (i.e. src has more), then 
      // propogate src's points to info to dst
      //if (srcReprID != dstReprID) {
      if (srcReprID != dstReprID && 
          !result[srcReprID].isSubsetEq(result[dstReprID])) {

        galois::gDebug("unifying ", dstReprID, " by ", srcReprID);
        // this getData call only exists to lock the node while in this scope
        graph.getData(nodes[srcReprID], flag);
        graph.getData(nodes[dstReprID], flag);
        // newPtsTo is positive if changes are made
        newPtsTo += result[dstReprID].unify(result[srcReprID]);
      } else {
        if (!result[srcReprID].isSubsetEq(result[dstReprID])) {
          std::abort();
        }
      }
    }

    return newPtsTo;
  }

 public:
  PTA(void) : ocd(*this) { }

  /**
   * Given the number of nodes in the constraint graph, initialize the
   * structures needed for the points-to algorithm.
   *
   * @param n Number of nodes in the constraint graph
   */
  void initialize(size_t n) {
    numNodes = n;
    result.resize(numNodes);
    nodes.resize(numNodes);
    ocd.init();

    unsigned nodeid = 0;

    // initialize vector of SparseBitVectors
    for (auto ii = result.begin(); ii != result.end(); ++ii, ++nodeid) {
      ii->init();

      GNode src = graph.createNode(Node(nodeid));
      graph.addNode(src);
      nodes[nodeid] = src;
    }
  }

  /**
   * TODO
   */
  void checkReprPointsTo() {
    for (unsigned ii = 0; ii < result.size(); ++ii) {
      unsigned repr = ocd.getFinalRepresentative(ii);
      if (repr != ii && !result[ii].isSubsetEq(result[repr])) {
        galois::gPrint("ERROR: pointsto(", ii, ") is not less than its "
                       "representative pointsto(", repr, ").\n");
      }
    }
  }

  /**
   * TODO
   */
  unsigned countPointsToFacts() {
    unsigned count = 0;
    for (auto ii = result.begin(); ii != result.end(); ++ii) {
      unsigned repr = ocd.getFinalRepresentative(ii - result.begin());
      count += result[repr].count();
    }
    return count;
  }

  /**
   * TODO
   */
  void printPointsToInfo() {
    std::string prefix = "v";
    for (auto ii = result.begin(); ii != result.end(); ++ii) {
      std::cerr << prefix << ii - result.begin() << ": ";
      unsigned repr = ocd.getFinalRepresentative(ii - result.begin());
      result[repr].print(std::cerr, prefix);
    }
  }


  /**
   * Run points-to-analysis without galois::for_each on a single thread.
   */
  void runSerial() {
    galois::gDebug("no of addr+copy constraints = ", 
                   addressCopyConstraints.size(), 
                   ", no of load+store constraints = ", 
                   loadStoreConstraints.size());
    galois::gDebug("no of nodes = ", nodes.size());

    std::deque<GNode> updates;
    updates = processAddressOfCopy<std::deque<GNode>>(addressCopyConstraints);
    processLoadStore(loadStoreConstraints, updates, 
                     galois::MethodFlag::UNPROTECTED);

    unsigned numIterations = 0;
    unsigned numUps = 0;

    // FIFO
    while (!updates.empty()) {
      galois::gDebug("Iteration ", numIterations++, ", updates.size=", 
                     updates.size(), "\n");
      GNode src = updates.front();
      updates.pop_front();

      galois::gDebug("processing updates element ", 
                     graph.getData(src, galois::MethodFlag::UNPROTECTED).id);

      for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
        GNode dst = graph.getEdgeDst(ii);

        unsigned newPtsTo = propagate(src, dst, galois::MethodFlag::UNPROTECTED);

        if (newPtsTo) {
          updates.push_back(dst);
        }
      }

      if (++numUps >= THRESHOLD_LS || updates.empty()) {
        //galois::gPrint("No of points-to facts computed = ", 
        //               countPointsToFacts(), "\n");
        numUps = 0;

        // After propagating all constraints, see if load/store
        // constraints need to be added in since graph was potentially updated
        //galois::gPrint(updates.size(), " us1\n");
        processLoadStore(loadStoreConstraints, updates);
        //galois::gPrint(updates.size(), " us2\n");
        // do cycle squashing
        if (updates.size() > THRESHOLD_OCD) {
          ocd.process(updates);
        }
      }
    }
  }

  /**
   * Run points-to-analysis using galois::for_each as the main loop.
   */
  void runParallel() {
    UpdatesVec updates;
    updates = processAddressOfCopy<UpdatesVec>(addressCopyConstraints);
    processLoadStore(loadStoreConstraints, updates, 
                     galois::MethodFlag::UNPROTECTED);

    while (!updates.empty()) {
      galois::for_each(
        galois::iterate(updates),
        [this] (const GNode& req, auto& ctx) {
          GNode src = req;

          for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
            GNode dst = graph.getEdgeDst(ii);
            unsigned newPtsTo = this->propagate(src, dst, 
                                                galois::MethodFlag::WRITE);
            if (newPtsTo) {
              ctx.push(dst);
            }
          }
        },
        galois::loopname("MainUpdateLoop"),
        galois::wl<galois::worklists::dChunkedFIFO<8>>()
      );
      galois::gPrint("No of points-to facts computed = ", 
                     countPointsToFacts(), "\n");

      updates.clear();

      // After propagating all constraints, see if load/store constraints need 
      // to be added in since graph was potentially updated
      //galois::gPrint(updates.size(), " us1\n");
      processLoadStore(loadStoreConstraints, updates, 
                       galois::MethodFlag::UNPROTECTED);
      //galois::gPrint(updates.size(), " us2\n");
      // do cycle squashing
      if (updates.size() > THRESHOLD_OCD) {
        ocd.process(updates);
      }
    }
  }

  /**
   * Read a constraint file and load its contents into memory.
   *
   * @param file filename to read
   * @returns number of nodes in the constraint graph
   */
  unsigned readConstraints(const char *file) {
    unsigned numNodes = 0;
    unsigned nconstraints = 0;

    std::ifstream cfile(file);
    std::string cstr;

    getline(cfile, cstr);   // # of vars.
    sscanf(cstr.c_str(), "%d", &numNodes);

    getline(cfile, cstr);   // # of constraints.
    sscanf(cstr.c_str(), "%d", &nconstraints);

    addressCopyConstraints.clear(); 
    loadStoreConstraints.clear();

    unsigned constraintNum;
    unsigned src;
    unsigned dst;
    unsigned offset;

    PtsToCons::ConstraintType type;

    // Create constraint objects and save them to appropriate location
    for (unsigned ii = 0; ii < nconstraints; ++ii) {
      getline(cfile, cstr);
      union { int as_int; PtsToCons::ConstraintType as_ctype; } type_converter;
      sscanf(cstr.c_str(), "%d,%d,%d,%d,%d", 
             &constraintNum, &src, &dst, &type_converter.as_int, &offset);

      type = type_converter.as_ctype;

      PtsToCons cc(type, src, dst);
      if (type == PtsToCons::AddressOf || type == PtsToCons::Copy) {
        addressCopyConstraints.push_back(cc);
      } else if (type == PtsToCons::Load || PtsToCons::Store) {
        loadStoreConstraints.push_back(cc);
      }
    }

    cfile.close();

    return numNodes;
  }


  /**
   * Prints the constraints in the passed in vector of constraints.
   *
   * @param constraints vector of PtsToCons
   */
  void printConstraints(PointsToConstraints &constraints) {
    for (auto ii = constraints.begin(); ii != constraints.end(); ++ii) {
      ii->print();
    }
  }
}; // end class PTA

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  PTA pta;

  size_t numNodes = pta.readConstraints(input.c_str());

  pta.initialize(numNodes);

  unsigned numThreads = galois::getActiveThreads();

  galois::StatTimer T;

  T.start();
  if (!useSerial) {
    galois::gPrint("-------- Parallel version: ", numThreads, " threads.\n");
    pta.runParallel();
  } else {
    galois::gPrint("-------- Sequential version.\n");
    pta.runSerial();
  }
  T.stop();

  galois::gPrint("No of points-to facts computed = ", 
                 pta.countPointsToFacts(), "\n");
  pta.checkReprPointsTo();
  //pta.printPointsToInfo();
  /*if (!skipVerify && !verify(result)) {
    std::cerr << "If graph was connected, verification failed\n";
    assert(0 && "If graph was connected, verification failed");
    abort();
  }*/

  return 0;
}
