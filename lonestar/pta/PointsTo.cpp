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
 * @author Loc Hoang <l_hoang@utexas.edu> (mainly documentation/cleanup)
 */
#include "galois/Galois.h"
#include "galois/graphs/Graph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include <fstream>
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

// no of nodes to be processed before adding load/store edges.
const unsigned THRESHOLD_LOADSTORE = 500;	
const unsigned THRESHOLD_OCD = 500;

////////////////////////////////////////////////////////////////////////////////
// Declaration of strutures, types, and variables
////////////////////////////////////////////////////////////////////////////////

/**
 * Structure of a node in the graph.
 */
struct Node {
	unsigned id;
	unsigned priority;

	Node() {
		id = 0;
		priority = 0;
	}
	Node(unsigned lid) : id(lid) {
		priority = 0;
	}
};

using Graph = galois::graphs::FirstGraph<Node, void, true>;
using GNode = Graph::GraphNode;

/** 
 * Structure representing an update request to the graph.
 *
 * copied from Andrew's SSSP. 
 **/
struct UpdateRequest {
  GNode n;
  unsigned int w; // this is the priority of the request

  UpdateRequest(GNode& N, unsigned int W)
    : n(N), w(W) {}
};

using UpdatesVec = std::vector<UpdateRequest>;

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
  using PointsToConstraints = std::vector<PtsToCons>;
  using PointsToInfo = std::vector<galois::SparseBitVector>;

 private:
  /** 
   * Online Cycle Detection and elimination structure + functions.
   */
  struct OnlineCycleDetection {	
   private:
    std::vector<unsigned> ancestors;
    std::vector<bool> visited;
    std::vector<unsigned> representative;
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
      nodeid = getFinalRepresentative(nodeid);

      // if the node is an ancestor, that means there's a path from the ancestor
      // to the ancestor (i.e. cycle)
      if (isAncestor(nodeid)) {
        cyclenode = nodeid;
        return true;
      }

      if (visited[nodeid]) {
        return false;
      }

      visited[nodeid] = true;

      // keep track of the current depth first search path
      ancestors.push_back(nodeid);

      GNode nn = outer.nodes[nodeid];

      for (auto ii : outer.graph.edges(nn, galois::MethodFlag::UNPROTECTED)) {
        // get id of destination node
        Node& nn = outer.graph.getData(outer.graph.getEdgeDst(ii), 
                                       galois::MethodFlag::UNPROTECTED);
        unsigned iiid = nn.id;

        // recursive depth first cycle detection; if a cycle is found,
        // collapse the path
        if (cycleDetect(iiid, cyclenode)) {
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
      unsigned reprrepr = getFinalRepresentative(repr);
      for (auto ii = ancestors.begin(); ii != ancestors.end(); ++ii) {
        if (*ii == repr) {
          galois::gDebug("collapsing cycle for ", repr);
          // cycle exists between nodes ancestors[*ii..end].
          for (auto jj = ii; jj != ancestors.end(); ++jj) {
            unsigned jjrepr = getFinalRepresentative(*jj);	// jjrepr has no representative.
            makeRepr(jjrepr, reprrepr);
          }
          break;
        }
      }
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

        if (!outer.result[repr].isSubsetEq(outer.result[nodeid])) {
          //outer.graph.getData(nodes[repr]);	// lock it.
          outer.result[repr].unify(outer.result[nodeid]);
        }
      }
    }
   public:
    PTA& outer;

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
      unsigned lnnid = nodeid;

      // Follow chain of representatives until a "root" is reached
      while (representative[lnnid] != NoRepresentative) {
        lnnid = representative[lnnid];
      }

      // path compression; make all things along path to final representative
      // point to the final representative
      unsigned repr = representative[nodeid];

      while (repr != NoRepresentative) {
        representative[nodeid] = lnnid;
        nodeid = repr;
        repr = representative[nodeid];
      }

      return lnnid;
    }

    /**
     * Go over all sources of new edges to see if there are cycles in them.
     * If so, collapse the cycles.
     *
     * @param updates vector of nodes that are sources of new edges.
     */
    void process(const UpdatesVec& updates) {	
      for (unsigned ii = 0; ii < outer.numNodes; ++ii) {
        visited[ii] = false;
      }

      unsigned cyclenode = NoRepresentative;	// set to invalid id.

      for (const UpdateRequest& up : updates) {
        Node &nn = outer.graph.getData(up.n, galois::MethodFlag::UNPROTECTED);
        unsigned nodeid = nn.id;
        galois::gDebug("cycle process ", nodeid);

        if (cycleDetect(nodeid, cyclenode)) {
          cycleCollapse(cyclenode);
        }
      }
    }
  }; // end struct OnlineCycleDetection

  OnlineCycleDetection ocd;
  Graph graph;
  std::vector<GNode> nodes;
  PointsToInfo result;
  PointsToConstraints addressCopyConstraints; 
  PointsToConstraints loadStoreConstraints;
  size_t numNodes = 0;

  void processLoadStoreSerial(PointsToConstraints &constraints, 
                              UpdatesVec &updates, 
                              galois::MethodFlag flag = 
                                  galois::MethodFlag::UNPROTECTED) {
    // add edges to the graph based on points-to information of the nodes
    // and then add the source of each edge to the updates.
    for (auto ii = constraints.begin(); ii != constraints.end(); ++ii) {
      //std::cout << "debug: Processing constraint: "; ii->print();
      unsigned src, dst;
      std::tie(src, dst) = ii->getSrcDst();

      unsigned srcrepr = ocd.getFinalRepresentative(src);
      unsigned dstrepr = ocd.getFinalRepresentative(dst);

      if (ii->getType() == PtsToCons::Load) {
        GNode &nndstrepr = nodes[dstrepr];
        std::vector<unsigned> ptsToOfSrc;
        result[srcrepr].getAllSetBits(ptsToOfSrc);
        for (auto pointee = ptsToOfSrc.begin(); pointee != ptsToOfSrc.end(); ++pointee) {
          unsigned pointeerepr = ocd.getFinalRepresentative(*pointee);
          if (pointeerepr != dstrepr && graph.findEdge(nodes[pointeerepr], nodes[dstrepr]) == graph.edge_begin(nodes[pointeerepr])) {
            GNode &nn = nodes[pointeerepr];
            graph.addEdge(nn, nndstrepr, flag);
            //std::cout << "debug: adding edge from " << *pointee << " to " << dst << std::endl;
            updates.push_back(UpdateRequest(nn, graph.getData(nn, galois::MethodFlag::UNPROTECTED).priority));
          }
        }
      } else {	// store.
        std::vector<unsigned> ptsToOfDst;
        bool newEdgeAdded = false;
        GNode &nnSrcRepr = nodes[srcrepr];
        result[dstrepr].getAllSetBits(ptsToOfDst);
        for (auto pointee = ptsToOfDst.begin(); pointee != ptsToOfDst.end(); ++pointee) {
          unsigned pointeerepr = ocd.getFinalRepresentative(*pointee);
          if (srcrepr != pointeerepr && graph.findEdge(nodes[srcrepr],nodes[pointeerepr]) == graph.edge_end(nodes[srcrepr])) {
            graph.addEdge(nnSrcRepr, nodes[pointeerepr], flag);
            //std::cout << "debug: adding edge from " << src << " to " << *pointee << std::endl;
            newEdgeAdded = true;
          }
        }
        if (newEdgeAdded) {
          updates.push_back(UpdateRequest(nnSrcRepr, graph.getData(nnSrcRepr, galois::MethodFlag::UNPROTECTED).priority));
        }
      }
    }
  }

  // add edges to the graph based on points-to information of the nodes
  // and then add the source of each edge to the updates.
  template <typename C>
  void processLoadStoreParallel(const PointsToConstraints &constraints, 
                                C& ctx, 
                                galois::MethodFlag flag = 
                                    galois::MethodFlag::WRITE) {
    for (auto& constraint : constraints) {
      //std::cout << "debug: Processing constraint: "; constraint->print();
      unsigned src, dst;
      std::tie(src, dst) = constraint.getSrcDst();

      unsigned srcRepr = ocd.getFinalRepresentative(src);
      unsigned dstRepr = ocd.getFinalRepresentative(dst);

      if (constraint.getType() == PtsToCons::Load) { 
        std::vector<unsigned> ptsToOfSrc;
        result[srcRepr].getAllSetBits(ptsToOfSrc);

        GNode &nnDstRepr = nodes[dstRepr]; // id of dst in graph

        for (unsigned pointee : ptsToOfSrc) {
          unsigned pointeeRepr = ocd.getFinalRepresentative(pointee);

          // add edge from pointee to dst if it doesn't already exist
          if (pointeeRepr != dstRepr && 
              graph.findEdge(nodes[pointeeRepr], nodes[dstRepr]) == 
                  graph.edge_end(nodes[pointeeRepr])) {
            GNode &nn = nodes[pointeeRepr];

            galois::gDebug("adding edge from ", pointee, " to ", dst);

            graph.addEdge(nn, nnDstRepr, flag);

            ctx.push(
              UpdateRequest(nn, 
                            graph.getData(nn, 
                                          galois::MethodFlag::WRITE).priority)
            );
          }
        }
      } else {	// store whatever src has into whatever dst points to
        std::vector<unsigned> ptsToOfDst;
        result[dstRepr].getAllSetBits(ptsToOfDst);
        GNode &nnSrcRepr = nodes[srcRepr];
        bool newEdgeAdded = false;

        for (unsigned pointee : ptsToOfDst) {
          unsigned pointeeRepr = ocd.getFinalRepresentative(pointee);

          // add edge from src -> pointee if it doesn't exist
          if (srcRepr != pointeeRepr && 
                graph.findEdge(nodes[srcRepr], nodes[pointeeRepr]) == 
                    graph.edge_end(nodes[srcRepr])) {
            galois::gDebug("adding edge from ", src, " to ", pointee);
            graph.addEdge(nnSrcRepr, nodes[pointeeRepr], flag);
            newEdgeAdded = true;
          }
        }

        if (newEdgeAdded) {
          ctx.push(
            UpdateRequest(nnSrcRepr, 
                          graph.getData(nnSrcRepr, 
                                        galois::MethodFlag::WRITE).priority)
          );
        }
      }
    }
  }

  /**
   * Processes the AddressOf and Copy constraints. 
   *
   * Sets the bitvector for AddressOf constraints, i.e. a set bit means
   * that you point to whtaever that bit represents.
   *
   * Creates edges for Copy constraints, i.e. edge from a to b indicates
   * b is a copy of a.
   *
   * @param constraints vector of AddressOf and Copy constraints
   * @returns vector of UpdatesRequests from all sources with new edges
   * added by the Copy constraint
   */
  UpdatesVec processAddressOfCopy(const PointsToConstraints& constraints) {
    UpdatesVec updates;

    for (auto ii = constraints.begin(); ii != constraints.end(); ++ii) {
      //std::cout << "debug: Processing constraint: "; ii->print();

      unsigned src;
      unsigned dst;
      std::tie(src, dst) = ii->getSrcDst();

      if (ii->getType() == PtsToCons::AddressOf) {
        if (result[dst].set(src)) { // this alters the state of the bitvector
          galois::gDebug("saving v", dst, "->v", src);
        }
      } else if (src != dst) {	// copy.
        GNode &nn = nodes[src];
        graph.addEdge(nn, nodes[dst], galois::MethodFlag::UNPROTECTED);
        galois::gDebug("adding edge from ", src, " to ", dst);

        updates.push_back(
          UpdateRequest(nn, 
                        graph.getData(nn, 
                                      galois::MethodFlag::UNPROTECTED).priority)
        );
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
   * @returns number changed in the dest bitvector (used to assign priority
   * in the worklist later)
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
      if (srcReprID != dstReprID && 
          !result[srcReprID].isSubsetEq(result[dstReprID])) {
        galois::gDebug("unifying ", dstReprID, " by ", srcReprID);
        // this getData call only exists to lock the node while in this scope
        graph.getData(nodes[dstReprID], flag);
        // newPtsTo is the number of chnages made to the bitvector
        newPtsTo = result[dstReprID].unify(result[srcReprID]);
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
      ii->init(numNodes);

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
        std::cout << "ERROR: pointsto(" << ii << ") is not less than its representative pointsto(" << repr << ").\n";
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
  void printPointsToInfo(PointsToInfo &result) {
    std::string prefix = "v";
    for (auto ii = result.begin(); ii != result.end(); ++ii) {
      std::cout << prefix << ii - result.begin() << ": ";
      unsigned repr = ocd.getFinalRepresentative(ii - result.begin());
      result[repr].print(std::cout, prefix);
    }
  }


  // TODO
  // This either hangs or is extremely slow even on the smallest constraint
  // file we have
  void runSerial() {
    //unsigned niteration = 0;
    //bool changed = false;
    unsigned nnodesprocessed = 0;

    UpdatesVec updates;
    updates = processAddressOfCopy(addressCopyConstraints);
    processLoadStoreSerial(loadStoreConstraints, updates, 
                           galois::MethodFlag::UNPROTECTED);	// required when there are zero copy constraints which keeps updates empty.

    //std::cout << "debug: no of addr+copy constraints = " << addressCopyConstraints.size() << ", no of load+store constraints = " << loadStoreConstraints.size() << std::endl;
    //std::cout << "debug: no of nodes = " << nodes.size() << std::endl;

    while (!updates.empty()) {
        //std::cout << "debug: Iteration " << ++niteration << ", updates.size=" << updates.size() << "\n"; 
      GNode src = updates.back().n;
      updates.pop_back();

      //std::cout << "debug: processing updates element " << graph.getData(src, galois::MethodFlag::UNPROTECTED).id << std::endl;
          for (auto ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
        GNode dst = graph.getEdgeDst(ii);
        unsigned newptsto = propagate(src, dst, galois::MethodFlag::UNPROTECTED);
        if (newptsto) {
          updates.push_back(UpdateRequest(dst, newptsto));
        }
      }
      if (++nnodesprocessed > THRESHOLD_LOADSTORE || updates.empty()) {
        nnodesprocessed = 0;
        processLoadStoreSerial(loadStoreConstraints, updates);	// add edges to graph, add their sources to updates.
        if (updates.size() > THRESHOLD_OCD) {
          ocd.process(updates);
        }
      }
    }
  }

  void runParallel() {
    UpdatesVec updates;
    //unsigned niteration = 0;
    
    updates = processAddressOfCopy(addressCopyConstraints);
    processLoadStoreSerial(loadStoreConstraints, updates, 
                           galois::MethodFlag::UNPROTECTED);

    //unsigned nfired = 0;
    //unsigned niter;

    galois::substrate::PerThreadStorage<unsigned> nfired;

    galois::on_each(
      [&] (unsigned tid, unsigned totalThreads) {
        *nfired.getLocal() = 0;
      }
    );

    // Lambda function that gets the key used by the OBIM worklist to order
    // things: grabs the priority of the UpdateRequest
    auto indexer = [] (const UpdateRequest& req) { return req.w; };
    using OBIM = galois::worklists::OrderedByIntegerMetric<
        decltype(indexer), 
        galois::worklists::dChunkedFIFO<1024>
    >;

    // TODO this is incorrect as processLoadStoreParallel may potentially
    // not be called at the very end of execution
    galois::for_each(
      galois::iterate(updates),
      [this, &nfired] (const UpdateRequest& req, auto& ctx) {
        if (++(*nfired.getLocal()) < THRESHOLD_LOADSTORE) {
          GNode src = req.n;

          //galois::gPrint("testing1\n");
          for (auto ii : graph.edges(src, galois::MethodFlag::UNPROTECTED)) {
            GNode dst = graph.getEdgeDst(ii);
            unsigned newPtsTo = this->propagate(src, dst, 
                                                galois::MethodFlag::WRITE);
            if (newPtsTo) {
              ctx.push(UpdateRequest(dst, newPtsTo));
            }
          }
        } else {
          *(nfired.getLocal()) = 0;

          this->processLoadStoreParallel(loadStoreConstraints, ctx, 
                                         galois::MethodFlag::WRITE);
          //if (wl.size() > THRESHOLD_OCD) {
          //  ocd.process(wl);
          //}
        }
        //if (nfired % 500 == 0) {
        //  std::cout << ++niter << std::endl;
        //}
      },
      galois::loopname("Main"),
      galois::wl<OBIM>(indexer)
    );
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

  // TODO 
  if (numThreads > 1) {
  	std::cout << "-------- Parallel version: " << numThreads << " threads.\n";
    pta.runParallel();
  } else {
  	std::cout << "-------- Sequential version.\n";
    pta.runSerial();
  }
  T.stop();


	//std::cout << "No of points-to facts computed = " << countPointsToFacts() << std::endl;
	//checkReprPointsTo();
	//printPointsToInfo(result);
  /*if (!skipVerify && !verify(result)) {
    std::cerr << "If graph was connected, verification failed\n";
    assert(0 && "If graph was connected, verification failed");
    abort();
  }*/

  return 0;
}
