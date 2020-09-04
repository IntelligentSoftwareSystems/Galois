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

#include "galois/Galois.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include <iostream>
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

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<bool>
    useSerial("serial",
              cll::desc("Runs serial version of the algorithm "
                        "(i.e. 1 thread, no galois::for_each) "
                        "(default false)"),
              cll::init(false));

static cll::opt<bool>
    printAnswer("printAnswer",
                cll::desc("If set, prints all points to facts "
                          "at the end"),
                cll::init(false));

static cll::opt<bool>
    useCycleDetection("ocd",
                      cll::desc("If set, online cycle detection is"
                                " used in algorithm (serial only) "
                                "(default false)"),
                      cll::init(false));

static cll::opt<unsigned>
    THRESHOLD_LS("lsThreshold",
                 cll::desc("Determines how many constraints to "
                           "process before load/store constraints "
                           "are reprocessed (serial only) "
                           "(default 500000)"),
                 cll::init(500000));

////////////////////////////////////////////////////////////////////////////////
// Declaration of strutures, types, and variables
////////////////////////////////////////////////////////////////////////////////

/**
 * Class representing a points-to constraint.
 */
class PtsToCons {
public:
  using ConstraintType = enum { AddressOf = 0, Copy, Load, Store, GEP };

private:
  unsigned src;
  unsigned dst;
  ConstraintType type;

public:
  PtsToCons(ConstraintType tt, unsigned ss, unsigned dd) {
    src  = ss;
    dst  = dd;
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
  ConstraintType getType() const { return type; }

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

    std::cerr << "\n";
  }
};

/**
 * Points to analysis runner base class. Does not have a run method itself.
 *
 * @tparam IsConcurrent if set to true, the data structures used for points
 * to results and outgoing edges will be thread safe
 */
template <bool IsConcurrent>
class PTABase {
  // sparse bit vector is concurrent or serial based on template parameter
  using SparseBitVector = galois::SparseBitVector<IsConcurrent>;

  using PointsToConstraints = std::vector<PtsToCons>;
  using PointsToInfo        = std::vector<SparseBitVector>;
  using EdgeVector          = std::vector<SparseBitVector>;

  using NodeAllocator =
      galois::FixedSizeAllocator<typename SparseBitVector::Node>;

protected:
  PointsToInfo pointsToResult; // pointsTo results for nodes
  EdgeVector outgoingEdges;    // holds outgoing edges of a node

  PointsToConstraints addressCopyConstraints;
  PointsToConstraints loadStoreConstraints;

  size_t numNodes = 0;

  ////////////////////////////////////////////////////////////////////////////////
  /**
   * Online Cycle Detection and elimination structure + functions.
   */
  struct OnlineCycleDetection {
  private:
    PTABase<IsConcurrent>&
        outerPTA; // reference to outer PTA instance to get runtime info

    galois::gstl::Vector<unsigned> ancestors; // TODO find better representation
    galois::gstl::Vector<bool> visited;       // TODO use better representation
    galois::gstl::Vector<unsigned> representative;

    unsigned NoRepresentative; // "constant" that represents no representative

    /**
     * @returns true of the nodeid is an ancestor node
     * FIXME find better way to do this instead of linear scan
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
    bool cycleDetect(unsigned nodeID, unsigned& cycleNode) {
      unsigned nodeRep = getFinalRepresentative(nodeID);

      // if the node is an ancestor, that means there's a path from the ancestor
      // to the ancestor (i.e. cycle)
      if (isAncestor(nodeRep)) {
        cycleNode = nodeRep;
        return true;
      }

      if (visited[nodeRep]) {
        return false;
      }

      visited[nodeRep] = true;

      // keep track of the current depth first search path
      ancestors.push_back(nodeRep);

      // don't use an iterator here because outgoing edges might get updated
      // during this loop
      std::vector<unsigned> repOutgoingEdges =
          outerPTA.outgoingEdges[nodeRep].getAllSetBits();

      for (auto dst : repOutgoingEdges) {
        // recursive depth first cycle detection; if a cycle is found,
        // collapse the path
        if (cycleDetect(dst, cycleNode)) {
          cycleCollapse(cycleNode);
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
            // jjRepr has no representative.
            unsigned jjRepr = getFinalRepresentative(*jj);
            makeRepr(jjRepr, repToChangeTo);
          }

          break;
        }
      }
    }

    /**
     * Make repr the representative of nodeID.
     *
     * @param nodeID node to change the representative of
     * @param repr nodeID will have its representative changed to this
     */
    void makeRepr(unsigned nodeID, unsigned repr) {
      if (repr != nodeID) {
        galois::gDebug("change repr[", nodeID, "] = ", repr);

        representative[nodeID] = repr;

        // the representative needs to have all of the items that the nodes
        // it is representing has, so if the node has more than the rep,
        // unify
        if (!outerPTA.pointsToResult[nodeID].isSubsetEq(
                outerPTA.pointsToResult[repr])) {
          outerPTA.pointsToResult[repr].unify(outerPTA.pointsToResult[nodeID]);
        }

        // unify edges as well if necessary since rep represents it now
        if (!outerPTA.outgoingEdges[nodeID].isSubsetEq(
                outerPTA.outgoingEdges[repr])) {
          outerPTA.outgoingEdges[repr].unify(outerPTA.outgoingEdges[nodeID]);
        }
      }
    }

  public:
    OnlineCycleDetection(PTABase<IsConcurrent>& o) : outerPTA(o) {}

    /**
     * Init fields (outerPTA needs to have numNodes set).
     */
    void init() {
      NoRepresentative = outerPTA.numNodes;
      visited.resize(outerPTA.numNodes);
      representative.resize(outerPTA.numNodes);

      for (unsigned ii = 0; ii < outerPTA.numNodes; ++ii) {
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

      // follow chain of representatives until a "root" is reached
      while (representative[finalRep] != NoRepresentative) {
        finalRep = representative[finalRep];
      }

      // path compression; make all things along path to final representative
      // point to the final representative
      unsigned curRep = representative[nodeid];

      while (curRep != NoRepresentative) {
        representative[nodeid] = finalRep;
        nodeid                 = curRep;
        curRep                 = representative[nodeid];
      }

      return finalRep;
    }

    /**
     * Go over all sources of new edges to see if there are cycles in them.
     * If so, collapse the cycles.
     *
     * @param updates vector of nodes that are sources of new edges.
     */
    template <typename VecType>
    void process(VecType& updates) {
      if (!useCycleDetection) {
        return;
      }

      // TODO this can probably be made more efficient (fill?)
      for (unsigned ii = 0; ii < outerPTA.numNodes; ++ii) {
        visited[ii] = false;
      }

      unsigned cycleNode = NoRepresentative; // set to invalid id.

      for (unsigned update : updates) {
        galois::gDebug("cycle process ", update);

        if (cycleDetect(update, cycleNode)) {
          cycleCollapse(cycleNode);
        }
      }
    }
  }; // end struct OnlineCycleDetection
  ////////////////////////////////////////////////////////////////////////////////

  OnlineCycleDetection ocd; // cycle detector/squasher; only works with serial

  /**
   * Adds edges to the graph based on load/store constraints.
   *
   * A load from src -> dst means anything that src points to must also
   * point to dst.
   *
   * A store from src -> dst means src must point to anything
   * that dst points to.
   *
   * Any updated nodes are returned in the updates vector.
   *
   * @tparam LoopInvoker Functor that will run the loop
   * @tparam VecType object that supports a push_back function that represents
   * nodes to be worked on
   *
   * @param constraints Load/store constraints to use to add edges
   * @param updates output variable that will have updated nodes added to it
   */
  template <typename LoopInvoker, typename VecType>
  void processLoadStore(const PointsToConstraints& constraints,
                        VecType& updates) {

    LoopInvoker()(galois::iterate(constraints), [&](auto constraint) {
      unsigned src;
      unsigned dst;
      std::tie(src, dst) = constraint.getSrcDst();

      unsigned srcRepr = ocd.getFinalRepresentative(src);
      unsigned dstRepr = ocd.getFinalRepresentative(dst);

      if (constraint.getType() == PtsToCons::Load) {
        for (auto pointee = pointsToResult[srcRepr].begin();
             pointee != pointsToResult[srcRepr].end(); pointee++) {
          unsigned pointeeRepr = ocd.getFinalRepresentative(*pointee);

          // add edge from pointee to dst if it doesn't already exist
          if (pointeeRepr != dstRepr &&
              !outgoingEdges[pointeeRepr].test(dstRepr)) {
            outgoingEdges[pointeeRepr].set(dstRepr);

            updates.push_back(pointeeRepr);
          }
        }
      } else { // store whatever src has into whatever dst points to
        bool newEdgeAdded = false;

        for (auto pointee = pointsToResult[dstRepr].begin();
             pointee != pointsToResult[dstRepr].end(); pointee++) {
          unsigned pointeeRepr = ocd.getFinalRepresentative(*pointee);

          // add edge from src -> pointee if it doesn't exist
          if (srcRepr != pointeeRepr &&
              !outgoingEdges[srcRepr].test(pointeeRepr)) {
            outgoingEdges[srcRepr].set(pointeeRepr);

            newEdgeAdded = true;
          }
        }

        if (newEdgeAdded) {
          updates.push_back(srcRepr);
        }
      }
    });
  }

  /**
   * Processes the AddressOf, Copy constraints.
   *
   * Sets the bitvector for AddressOf constraints, i.e. a set bit means
   * that you point to whatever that bit represents.
   *
   * Creates edges for Copy constraints, i.e. edge from a to b indicates
   * b is a copy of a.
   *
   * @tparam LoopInvoker Functor that will run the loop
   * @tparam VecType object that supports a push_back function as well
   * as iteration over pushed objects
   *
   * @param constraints vector of AddressOf and Copy constraints
   * @returns vector of UpdatesRequests from all sources with new edges
   * added by the Copy constraint
   */
  template <typename LoopInvoker, typename VecType>
  VecType processAddressOfCopy(const PointsToConstraints& constraints) {
    VecType updates;

    LoopInvoker()(galois::iterate(constraints), [&](auto ii) {
      unsigned src;
      unsigned dst;

      std::tie(src, dst) = ii.getSrcDst();

      if (ii.getType() == PtsToCons::AddressOf) { // addressof; save point info
        pointsToResult[dst].set(src);
      } else if (src != dst) { // copy constraint; add an edge
        outgoingEdges[src].set(dst);
        updates.push_back(src);
      }
    });

    return updates;
  }

  /**
   * If an edge exists from src to dst, then dst is a copy of src.
   * Propogate any points to information from source to dest.
   *
   * @param src Source node in graph
   * @param dst Dest node in graph
   * @returns non-negative value if any bitvector has changed
   */
  unsigned propagate(unsigned src, unsigned dst) {
    unsigned newPtsTo = 0;

    if (src != dst) {
      unsigned srcRepr = ocd.getFinalRepresentative(src);
      unsigned dstRepr = ocd.getFinalRepresentative(dst);

      // if src is a not subset of dst... (i.e. src has more), then
      // propogate src's points to info to dst
      if (srcRepr != dstRepr &&
          !pointsToResult[srcRepr].isSubsetEq(pointsToResult[dstRepr])) {
        // galois::gDebug("unifying ", dstRepr, " by ", srcRepr);
        // newPtsTo is positive if changes are made
        newPtsTo += pointsToResult[dstRepr].unify(pointsToResult[srcRepr]);
      }
    }

    return newPtsTo;
  }

public:
  PTABase() : ocd(*this) {}

  /**
   * Given the number of nodes in the constraint graph, initialize the
   * structures needed for the points-to algorithm.
   *
   * @param n Number of nodes in the constraint graph
   * @param nodeAllocator galois allocator object to allocate nodes in the
   * sparse bit vector
   */
  void initialize(size_t n, NodeAllocator& nodeAllocator) {
    numNodes = n;

    // initialize different constructs based on which version is being run
    pointsToResult.resize(numNodes);
    outgoingEdges.resize(numNodes);

    // initialize vectors
    for (unsigned i = 0; i < numNodes; i++) {
      pointsToResult[i].init(&nodeAllocator);
      outgoingEdges[i].init(&nodeAllocator);
    }

    ocd.init();
  }

  //! frees memory allocated by the node allocator
  void freeNodeAllocatorMemory() {
    for (unsigned i = 0; i < numNodes; i++) {
      pointsToResult[i].freeAll();
      outgoingEdges[i].freeAll();
    }
  }

  /**
   * Read a constraint file and load its contents into memory.
   *
   * @param file filename to read
   * @returns number of nodes in the constraint graph
   */
  unsigned readConstraints(const char* file) {
    galois::gInfo("GEP constraints (constraint type 4) and any constraints "
                  "with offsets are ignored.");

    unsigned numNodes     = 0;
    unsigned nconstraints = 0;

    std::ifstream cfile(file);
    std::string cstr;

    getline(cfile, cstr); // # of vars.
    sscanf(cstr.c_str(), "%d", &numNodes);

    getline(cfile, cstr); // # of constraints.
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
      union {
        int as_int;
        PtsToCons::ConstraintType as_ctype;
      } type_converter;
      sscanf(cstr.c_str(), "%d,%d,%d,%d,%d", &constraintNum, &src, &dst,
             &type_converter.as_int, &offset);

      type = type_converter.as_ctype;

      PtsToCons cc(type, src, dst);

      if (type == PtsToCons::AddressOf || type == PtsToCons::Copy) {
        addressCopyConstraints.push_back(cc);
      } else if (type == PtsToCons::Load || type == PtsToCons::Store) {
        if (offset == 0) { // ignore load/stores with offsets
          loadStoreConstraints.push_back(cc);
        }
      }
      // ignore GEP constraints
    }

    cfile.close();

    return numNodes;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Debugging/output functions
  //////////////////////////////////////////////////////////////////////////////

  /**
   * Prints the constraints in the passed in vector of constraints.
   *
   * @param constraints vector of PtsToCons
   */
  void printConstraints(PointsToConstraints& constraints) {
    for (auto ii = constraints.begin(); ii != constraints.end(); ++ii) {
      ii->print();
    }
  }

  /**
   * Checks to make sure that all representative point to at LEAST
   * what the nodes that it represents are pointing to. Necessary but not
   * sufficient check for correctness.
   */
  void checkReprPointsTo() {
    for (unsigned ii = 0; ii < pointsToResult.size(); ++ii) {
      unsigned repr = ocd.getFinalRepresentative(ii);
      if (repr != ii && !pointsToResult[ii].isSubsetEq(pointsToResult[repr])) {
        galois::gError("pointsto(", ii,
                       ") is not less than its "
                       "representative pointsto(",
                       repr, ").");
      }
    }
  }

  /**
   * Makes sure that the representative of a set of nodes has all of the
   * edges that the nodes it is representing has.
   */
  void checkReprEdges() {
    for (unsigned ii = 0; ii < outgoingEdges.size(); ++ii) {
      unsigned repr = ocd.getFinalRepresentative(ii);
      if (repr != ii && !outgoingEdges[ii].isSubsetEq(outgoingEdges[repr])) {
        galois::gError("edges(", ii,
                       ") is not less than its "
                       "representative edges(",
                       repr, ").");
      }
    }
  }

  /**
   * @returns The total number of points to facts in the system.
   */
  unsigned countPointsToFacts() {
    unsigned count = 0;

    for (auto ii = pointsToResult.begin(); ii != pointsToResult.end(); ++ii) {
      unsigned repr = ocd.getFinalRepresentative(ii - pointsToResult.begin());
      count += pointsToResult[repr].count();
    }

    return count;
  }

  /**
   * Prints out points to info for all verticies in the constraint graph.
   */
  void printPointsToInfo() {
    std::string prefix = "v";

    for (auto ii = pointsToResult.begin(); ii != pointsToResult.end(); ++ii) {
      std::cerr << prefix << ii - pointsToResult.begin() << ": ";
      unsigned repr = ocd.getFinalRepresentative(ii - pointsToResult.begin());
      pointsToResult[repr].print(std::cerr, prefix);
    }
  }
}; // end class PTA

/**
 * Serial points to executor.
 */
class PTASerial : public PTABase<false> {
public:
  /**
   * Run points-to-analysis on a single thread.
   */
  void run() {
    galois::gDebug(
        "no of addr+copy constraints = ", addressCopyConstraints.size(),
        ", no of load+store constraints = ", loadStoreConstraints.size());
    galois::gDebug("no of nodes = ", numNodes);

    std::deque<unsigned> updates;
    updates = processAddressOfCopy<galois::StdForEach, std::deque<unsigned>>(
        addressCopyConstraints);
    processLoadStore<galois::StdForEach>(loadStoreConstraints, updates);

    unsigned numUps = 0;

    // FIFO
    while (!updates.empty()) {
      unsigned src = updates.front();
      updates.pop_front();

      for (auto dst = outgoingEdges[src].begin();
           dst != outgoingEdges[src].end(); dst++) {
        unsigned newPtsTo = propagate(src, *dst);

        if (newPtsTo) { // newPtsTo is positive if dst changed
          updates.push_back(ocd.getFinalRepresentative(*dst));
        }

        numUps++;
      }

      if (updates.empty() || numUps >= THRESHOLD_LS) {
        galois::gDebug("No of points-to facts computed = ",
                       countPointsToFacts());
        numUps = 0;

        // After propagating all constraints, see if load/store
        // constraints need to be added in since graph was potentially updated
        processLoadStore<galois::StdForEach>(loadStoreConstraints, updates);

        // do cycle squashing
        ocd.process(updates);
      }
    }
  }
};

/**
 * Concurrent points to executor.
 */
class PTAConcurrent : public PTABase<true> {
public:
  /**
   * Run points-to-analysis using galois::for_each as the main loop.
   */
  void run() {
    galois::gDebug(
        "no of addr+copy constraints = ", addressCopyConstraints.size(),
        ", no of load+store constraints = ", loadStoreConstraints.size());
    galois::gDebug("no of nodes = ", numNodes);

    galois::InsertBag<unsigned> updates;
    updates = processAddressOfCopy<galois::DoAll, galois::InsertBag<unsigned>>(
        addressCopyConstraints);
    processLoadStore<galois::DoAll>(loadStoreConstraints, updates);

    while (!updates.empty()) {
      galois::for_each(
          galois::iterate(updates),
          [this](unsigned req, auto& ctx) {
            for (auto dst = this->outgoingEdges[req].begin();
                 dst != this->outgoingEdges[req].end(); dst++) {
              unsigned newPtsTo = this->propagate(req, *dst);

              if (newPtsTo)
                ctx.push(this->ocd.getFinalRepresentative(*dst));
            }
          },
          galois::loopname("PointsToMainUpdateLoop"),
          galois::disable_conflict_detection(),
          galois::wl<galois::worklists::PerSocketChunkFIFO<8>>());

      galois::gDebug("No of points-to facts computed = ", countPointsToFacts());

      updates.clear();

      // After propagating all constraints, see if load/store constraints need
      // to be added in since graph was potentially updated
      processLoadStore<galois::DoAll>(loadStoreConstraints, updates);

      // do cycle squashing
      // ocd.process(updates); // TODO have parallel OCD, if possible
    }
  }
};

/**
 * Method from running PTA.
 */
template <typename PTAClass, typename Alloc>
void runPTA(PTAClass& pta, Alloc& nodeAllocator) {
  size_t numNodes = pta.readConstraints(inputFile.c_str());
  pta.initialize(numNodes, nodeAllocator);

  galois::StatTimer execTime("Timer_0");

  execTime.start();
  pta.run();
  execTime.stop();

  galois::gInfo("No of points-to facts computed = ", pta.countPointsToFacts());

  if (!skipVerify) {
    galois::gInfo("Doing verification step");
    pta.checkReprPointsTo();
    pta.checkReprEdges();
  }

  if (printAnswer) {
    pta.printPointsToInfo();
  }

  // free everything nodeallocator allocated
  pta.freeNodeAllocatorMemory();
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, nullptr, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  // depending on serial or concurrent, create the correct class and pass it
  // into the run harness which takes care of the rest
  if (!useSerial) {
    galois::gInfo("-------- Parallel version: ", galois::getActiveThreads(),
                  " threads.");
    galois::gInfo("Note correctness of this version is relative to the serial "
                  "version.");

    PTAConcurrent p;
    galois::FixedSizeAllocator<typename galois::SparseBitVector<true>::Node>
        nodeAllocator;
    runPTA(p, nodeAllocator);
  } else {
    galois::gInfo("-------- Sequential version.");
    galois::gInfo(
        "The load store threshold (-lsThreshold) may need tweaking for "
        "best performance; its current setting may not be the best for "
        "your input and may actually degrade performance.");
    PTASerial p;
    galois::FixedSizeAllocator<typename galois::SparseBitVector<false>::Node>
        nodeAllocator;
    runPTA(p, nodeAllocator);
  }

  totalTime.stop();

  return 0;
}
