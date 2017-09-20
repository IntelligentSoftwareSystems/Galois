/** Points-to Analysis application -*- C++ -*-
 * @file
 *
 * An inclusion-based points-to analysis algorithm to demostrate the Galois system.
 *
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
 * @author rupesh nasre. <rupesh0508@gmail.com>
 */
#include "galois/Galois.h"
#include "galois/Bag.h"
#include "SparseBitVector.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/FileGraph.h"
#include "galois/worklists/WorkList.h"
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <stack>

namespace cll = llvm::cl;

namespace {

const char* name = "Points-to Analysis";
const char* desc = "Performs inclusion-based points-to analysis over the input constraints.";
const char* url = NULL;

static cll::opt<std::string> input(cll::Positional, cll::desc("<constraints>"), cll::Required);

const unsigned THRESHOLD_LOADSTORE = 500;	// no of nodes to be processed before adding load/store edges.
const unsigned THRESHOLD_OCD = 500;

struct Node {
	unsigned id;
	unsigned priority;

	Node() {
		id = 0;
		priority = 0;
	}
	Node(unsigned lid): id(lid) {
		priority = 0;
	}
};

typedef galois::graphs::FirstGraph<Node,void,true> Graph;
typedef Graph::GraphNode GNode;

/* copied from Andrew's SSSP. */

struct UpdateRequest {
  GNode n;
  unsigned int w;

  UpdateRequest(GNode& N, unsigned int W)
    :n(N), w(W)
  {}
};

struct UpdateRequestIndexer
  : std::binary_function<UpdateRequest, unsigned int, unsigned int> {
  unsigned int operator() (const UpdateRequest& val) const {
    unsigned int t = val.w;
    return t;
  }
};


//typedef std::pair<GNode,GNode> Edge;
typedef std::vector<UpdateRequest> WorkList;
typedef galois::worklists::OrderedByIntegerMetric<UpdateRequestIndexer, galois::worklists::dChunkedFIFO<1024> > OBIM;

class PtsToCons {
public:
	typedef enum {AddressOf = 0, Copy, Load, Store} ConstraintType;

	PtsToCons(ConstraintType tt, unsigned ss, unsigned dd) {
		src = ss;
		dst = dd;
		type = tt;
	}
	void getSrcDst(unsigned &ss, unsigned &dd) {
		ss = src;
		dd = dst;
	}
	ConstraintType getType() {
		return type;
	}
	void print() {
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
private:
	unsigned src, dst;
	ConstraintType type;
};

typedef std::vector<PtsToCons> PointsToConstraints;
typedef std::vector<galois::SparseBitVector> PointsToInfo;

Graph graph;
std::vector<GNode> nodes;
PointsToInfo result;
PointsToConstraints addrcopyconstraints, loadstoreconstraints;
unsigned numNodes = 0;

class OCD {	// Online Cycle Detection and elimination.
public:
  OCD() { }

  void init() {
	NoRepresentative = numNodes;
  	representative.resize(numNodes);
  	visited.resize(numNodes);
  	for (unsigned ii = 0; ii < numNodes; ++ii) {
		representative[ii] = NoRepresentative;
	}
  }
  void process(WorkList &worklist) {	// worklist of nodes that are sources of new edges.

  	for (unsigned ii = 0; ii < numNodes; ++ii) {
		visited[ii] = false;
	}
	unsigned cyclenode = numNodes;	// set to invalid id.
	for (worklists::iterator ii = worklist.begin(); ii != worklist.end(); ++ii) {
		Node &nn = graph.getData(ii->n, galois::MethodFlag::UNPROTECTED);
		unsigned nodeid = nn.id;
		//std::cout << "debug: cycle process " << nodeid << std::endl;
		if (cycleDetect(nodeid, cyclenode)) {
			cycleCollapse(cyclenode);
		}
	}
  }

  bool cycleDetect(unsigned nodeid, unsigned &cyclenode) {	// it is okay not to detect all cycles as it is only an efficiency concern.
  	nodeid = getFinalRepresentative(nodeid);
	if (isAncestor(nodeid)) {
		cyclenode = nodeid;
		return true;
	}
	if (visited[nodeid]) {
		return false;
	}
	visited[nodeid] = true;
	ancestors.push_back(nodeid);

	GNode nn = nodes[nodeid];
	for (Graph::edge_iterator ii = graph.edge_begin(nn, galois::MethodFlag::UNPROTECTED), ei = graph.edge_end(nn, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
	  Node &nn = graph.getData(graph.getEdgeDst(ii), galois::MethodFlag::UNPROTECTED);
		unsigned iiid = nn.id;
		if (cycleDetect(iiid, cyclenode)) {
			//return true;	// don't pop from ancestors.
			cycleCollapse(cyclenode);
		}
	}
	ancestors.pop_back();
	return false;
  }
  void cycleCollapse(unsigned repr) {
	// assert(repr is present in ancestors).
	//static unsigned ncycles = 0;
	unsigned reprrepr = getFinalRepresentative(repr);
	for (std::vector<unsigned>::iterator ii = ancestors.begin(); ii != ancestors.end(); ++ii) {
		if (*ii == repr) {
			//std::cout << "debug: collapsing cycle for " << repr << std::endl;
			// cycle exists between nodes ancestors[*ii..end].
			for (std::vector<unsigned>::iterator jj = ii; jj != ancestors.end(); ++jj) {
				unsigned jjrepr = getFinalRepresentative(*jj);	// jjrepr has no representative.
				makeRepr(jjrepr, reprrepr);
			}
			//std::cout << "debug: cycles collapsed = " << ++ncycles << std::endl;
			break;
		}
	}
	//ancestors.clear();	// since collapse is called from top level process(), the ancestors need to be cleared for the next element in the worklist.
  }
  void makeRepr(unsigned nodeid, unsigned repr) {
	// make repr the representative of nodeid.
	if (repr != nodeid) {
		//std::cout << "debug: repr[" << nodeid << "] = " << repr << std::endl;
		representative[nodeid] = repr;
		if (!result[repr].isSubsetEq(result[nodeid])) {
			//graph.getData(nodes[repr]);	// lock it.
			result[repr].unify(result[nodeid]);
		}
	}
  }
  unsigned getFinalRepresentative(unsigned nodeid) {
	unsigned lnnid = nodeid;
	while (representative[lnnid] != NoRepresentative) {
		lnnid = representative[lnnid];
	}
	// path compression.
	unsigned repr = representative[nodeid];
	while (repr != NoRepresentative) {
		representative[nodeid] = lnnid;
		nodeid = repr;
		repr = representative[nodeid];
	}
	return lnnid;
  }

private:
	bool isAncestor(unsigned nodeid) {
		for (std::vector<unsigned>::iterator ii = ancestors.begin(); ii != ancestors.end(); ++ii) {
			if (*ii == nodeid) {
				return true;
			}
		}
		return false;
	}
	std::vector<unsigned> ancestors;
	std::vector<bool> visited;
	std::vector<unsigned> representative;
	unsigned NoRepresentative;
};

OCD ocd;

void checkReprPointsTo() {
	for (unsigned ii = 0; ii < result.size(); ++ii) {
		unsigned repr = ocd.getFinalRepresentative(ii);
		if (repr != ii && !result[ii].isSubsetEq(result[repr])) {
			std::cout << "ERROR: pointsto(" << ii << ") is not less than its representative pointsto(" << repr << ").\n";
		}
	}
}

unsigned countPointsToFacts() {
	unsigned count = 0;
	for (PointsToInfo::iterator ii = result.begin(); ii != result.end(); ++ii) {
		unsigned repr = ocd.getFinalRepresentative(ii - result.begin());
		count += result[repr].count();
	}
	return count;
}
void printPointsToInfo(PointsToInfo &result) {
	std::string prefix = "v";
	for (PointsToInfo::iterator ii = result.begin(); ii != result.end(); ++ii) {
		std::cout << prefix << ii - result.begin() << ": ";
		unsigned repr = ocd.getFinalRepresentative(ii - result.begin());
		result[repr].print(std::cout, prefix);
	}
}
void processLoadStoreSerial(PointsToConstraints &constraints, WorkList &worklist, galois::MethodFlag flag = galois::MethodFlag::UNPROTECTED) {
	// add edges to the graph based on points-to information of the nodes
	// and then add the source of each edge to the worklist.
	for (PointsToConstraints::iterator ii = constraints.begin(); ii != constraints.end(); ++ii) {
		unsigned src, dst;
		//std::cout << "debug: Processing constraint: "; ii->print();
		ii->getSrcDst(src, dst);
		unsigned srcrepr = ocd.getFinalRepresentative(src);
		unsigned dstrepr = ocd.getFinalRepresentative(dst);
		if (ii->getType() == PtsToCons::Load) {
			GNode &nndstrepr = nodes[dstrepr];
			std::vector<unsigned> ptstoOfSrc;
			result[srcrepr].getAllSetBits(ptstoOfSrc);
			for (std::vector<unsigned>::iterator pointee = ptstoOfSrc.begin(); pointee != ptstoOfSrc.end(); ++pointee) {
				unsigned pointeerepr = ocd.getFinalRepresentative(*pointee);
				if (pointeerepr != dstrepr && graph.findEdge(nodes[pointeerepr], nodes[dstrepr]) == graph.edge_begin(nodes[pointeerepr])) {
					GNode &nn = nodes[pointeerepr];
					graph.addEdge(nn, nndstrepr, flag);
					//std::cout << "debug: adding edge from " << *pointee << " to " << dst << std::endl;
					worklist.push_back(UpdateRequest(nn, graph.getData(nn, galois::MethodFlag::UNPROTECTED).priority));
				}
			}
		} else {	// store.
			std::vector<unsigned> ptstoOfDst;
			bool newedgeadded = false;
			GNode &nnsrcrepr = nodes[srcrepr];
			result[dstrepr].getAllSetBits(ptstoOfDst);
			for (std::vector<unsigned>::iterator pointee = ptstoOfDst.begin(); pointee != ptstoOfDst.end(); ++pointee) {
				unsigned pointeerepr = ocd.getFinalRepresentative(*pointee);
				if (srcrepr != pointeerepr && graph.findEdge(nodes[srcrepr],nodes[pointeerepr]) == graph.edge_end(nodes[srcrepr])) {
					graph.addEdge(nnsrcrepr, nodes[pointeerepr], flag);
					//std::cout << "debug: adding edge from " << src << " to " << *pointee << std::endl;
					newedgeadded = true;
				}
			}
			if (newedgeadded) {
				worklist.push_back(UpdateRequest(nnsrcrepr, graph.getData(nnsrcrepr, galois::MethodFlag::UNPROTECTED).priority));
			}
		}
	}
}
void processAddressOfCopy(PointsToConstraints &constraints, WorkList &worklist) {
	for (PointsToConstraints::iterator ii = constraints.begin(); ii != constraints.end(); ++ii) {
		unsigned src, dst;
		//std::cout << "debug: Processing constraint: "; ii->print();
		ii->getSrcDst(src, dst);
		if (ii->getType() == PtsToCons::AddressOf) {
			if (result[dst].set(src)) {
				//std::cout << "debug: saving v" << dst << "->v" << src << std::endl;
			}
		} else if (src != dst) {	// copy.
			GNode &nn = nodes[src];
			graph.addEdge(nn, nodes[dst], galois::MethodFlag::UNPROTECTED);
			//std::cout << "debug: adding edge from " << src << " to " << dst << std::endl;
			worklist.push_back(UpdateRequest(nn, graph.getData(nn, galois::MethodFlag::UNPROTECTED).priority));
		}
	}
}
unsigned propagate(GNode &src, GNode &dst, galois::MethodFlag flag = galois::MethodFlag::WRITE) {
	unsigned srcid = graph.getData(src, galois::MethodFlag::UNPROTECTED).id;
	unsigned dstid = graph.getData(dst, galois::MethodFlag::UNPROTECTED).id;
	unsigned newptsto = 0;

	if (srcid != dstid) {
		unsigned srcreprid = ocd.getFinalRepresentative(srcid);
		unsigned dstreprid = ocd.getFinalRepresentative(dstid);
		if (srcreprid != dstreprid && !result[srcreprid].isSubsetEq(result[dstreprid])) {
			//std::cout << "debug: unifying " << dstreprid << " by " << srcreprid << std::endl;
			graph.getData(nodes[dstreprid], flag);
			newptsto = result[dstreprid].unify(result[srcreprid]);
			//newptsto = 0;
		}
	}
	return newptsto;
}

void processLoadStore(PointsToConstraints &constraints, WorkList &worklist, galois::MethodFlag flag = galois::MethodFlag::WRITE) {
	// add edges to the graph based on points-to information of the nodes
	// and then add the source of each edge to the worklist.
	for (PointsToConstraints::iterator ii = constraints.begin(); ii != constraints.end(); ++ii) {
		unsigned src, dst;
		//std::cout << "debug: Processing constraint: "; ii->print();
		ii->getSrcDst(src, dst);
		unsigned srcrepr = ocd.getFinalRepresentative(src);
		unsigned dstrepr = ocd.getFinalRepresentative(dst);
		if (ii->getType() == PtsToCons::Load) {
			GNode &nndstrepr = nodes[dstrepr];
			std::vector<unsigned> ptstoOfSrc;
			result[srcrepr].getAllSetBits(ptstoOfSrc);
			for (std::vector<unsigned>::iterator pointee = ptstoOfSrc.begin(); pointee != ptstoOfSrc.end(); ++pointee) {
				unsigned pointeerepr = ocd.getFinalRepresentative(*pointee);
				if (pointeerepr != dstrepr && graph.findEdge(nodes[pointeerepr], nodes[dstrepr]) == graph.edge_end(nodes[pointeerepr])) {
					GNode &nn = nodes[pointeerepr];
					graph.addEdge(nn, nndstrepr, flag);
					//std::cout << "debug: adding edge from " << *pointee << " to " << dst << std::endl;
					worklist.push_back(UpdateRequest(nn, graph.getData(nn, galois::MethodFlag::WRITE).priority));
				}
			}
		} else {	// store.
			std::vector<unsigned> ptstoOfDst;
			bool newedgeadded = false;
			GNode &nnsrcrepr = nodes[srcrepr];
			result[dstrepr].getAllSetBits(ptstoOfDst);
			for (std::vector<unsigned>::iterator pointee = ptstoOfDst.begin(); pointee != ptstoOfDst.end(); ++pointee) {
				unsigned pointeerepr = ocd.getFinalRepresentative(*pointee);
				if (srcrepr != pointeerepr && graph.findEdge(nodes[srcrepr],nodes[pointeerepr]) == graph.edge_end(nodes[srcrepr])) {
					graph.addEdge(nnsrcrepr, nodes[pointeerepr], flag);
					//std::cout << "debug: adding edge from " << src << " to " << *pointee << std::endl;
					newedgeadded = true;
				}
			}
			if (newedgeadded) {
				worklist.push_back(UpdateRequest(nnsrcrepr, graph.getData(nnsrcrepr, galois::MethodFlag::WRITE).priority));
			}
		}
	}
}

unsigned nfired;
//unsigned niter;
struct Process {
  Process() { }

  template<typename Context>
  void operator()(UpdateRequest &req, Context& ctx) {
   if (++nfired < THRESHOLD_LOADSTORE) {
    GNode &src = req.n;
    for (Graph::edge_iterator ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
        ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
      GNode dst = graph.getEdgeDst(ii);
	unsigned newptsto = propagate(src, dst, galois::MethodFlag::WRITE);
	if (newptsto) {
		ctx.push(UpdateRequest(dst, newptsto));
	}
    }
  } else {
	nfired = 0;
	WorkList wl;
	processLoadStore(loadstoreconstraints, wl, galois::MethodFlag::WRITE);
	/*if (wl.size() > THRESHOLD_OCD) {
		ocd.process(wl);
	}*/
	for (worklists::iterator ii = wl.begin(); ii != wl.end(); ++ii) {
		ctx.push(*ii);
	}
   }
   /*if (nfired % 500 == 0) {
	std::cout << ++niter << std::endl;
   }*/

  }
};

void runSerial(PointsToConstraints &addrcopyconstraints, PointsToConstraints &loadstoreconstraints) {
	WorkList worklist;
	//unsigned niteration = 0;
	//bool changed = false;
	unsigned nnodesprocessed = 0;

	processAddressOfCopy(addrcopyconstraints, worklist);
        processLoadStoreSerial(loadstoreconstraints, worklist, galois::MethodFlag::UNPROTECTED);	// required when there are zero copy constraints which keeps worklist empty.

	//std::cout << "debug: no of addr+copy constraints = " << addrcopyconstraints.size() << ", no of load+store constraints = " << loadstoreconstraints.size() << std::endl;
	//std::cout << "debug: no of nodes = " << nodes.size() << std::endl;

	while (!worklist.empty()) {
  		//std::cout << "debug: Iteration " << ++niteration << ", worklist.size=" << worklist.size() << "\n"; 
		GNode src = worklist.back().n;
		worklist.pop_back();

		//std::cout << "debug: processing worklist element " << graph.getData(src, galois::MethodFlag::UNPROTECTED).id << std::endl;
    		for (Graph::edge_iterator ii = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED), ei = graph.edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii) {
		  GNode dst = graph.getEdgeDst(ii);
			unsigned newptsto = propagate(src, dst, galois::MethodFlag::UNPROTECTED);
			if (newptsto) {
				worklist.push_back(UpdateRequest(dst, newptsto));
			}
		}
		if (++nnodesprocessed > THRESHOLD_LOADSTORE || worklist.empty()) {
			nnodesprocessed = 0;
	        	processLoadStoreSerial(loadstoreconstraints, worklist);	// add edges to graph, add their sources to worklist.
			if (worklist.size() > THRESHOLD_OCD) {
				ocd.process(worklist);
			}
		}
        }
}

void runParallel(PointsToConstraints &addrcopyconstraints, PointsToConstraints &loadstoreconstraints) {
	WorkList worklist;
	//unsigned niteration = 0;
	
	processAddressOfCopy(addrcopyconstraints, worklist);
	processLoadStore(loadstoreconstraints, worklist, galois::MethodFlag::UNPROTECTED);

		//using namespace galois::runtime::WorkList;
	  	//galois::for_each<LocalQueues<dChunkedFIFO<1024> > >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<FIFO<> >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<ChunkedFIFO<1024> >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<dChunkedFIFO<1024> >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<ChunkedLIFO<32> >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<LocalQueues<ChunkedLIFO<32> > >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<LocalQueues<FIFO<1024> > >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<LocalQueues<ChunkedFIFO<32> > >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<LocalQueues<dChunkedLIFO<32>, FIFO<> > >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<LocalQueues<ChunkedLIFO<1024>, FIFO<> > >(worklist.begin(), worklist.end(), Process());
	  	//galois::for_each<LocalQueues<ChunkedFIFO<1024>, FIFO<> > >(worklist.begin(), worklist.end(), Process());
	  	galois::for_each(worklist.begin(), worklist.end(), Process());
}

unsigned readConstraints(const char *file, PointsToConstraints &addrcopyconstraints, PointsToConstraints &loadstoreconstraints) {
	unsigned numNodes = 0;
	unsigned nconstraints = 0;

        std::ifstream cfile(file);
        std::string cstr;

        getline(cfile, cstr);   // no of vars.
        sscanf(cstr.c_str(), "%d", &numNodes);

        getline(cfile, cstr);   // no of constraints.
        sscanf(cstr.c_str(), "%d", &nconstraints);

        addrcopyconstraints.clear(); 
	loadstoreconstraints.clear();

	unsigned consno, src, dst, offset;
	PtsToCons::ConstraintType type;

        for (unsigned ii = 0; ii < nconstraints; ++ii) {
		getline(cfile, cstr);
                union { int as_int; PtsToCons::ConstraintType as_ctype; } type_converter;
		sscanf(cstr.c_str(), "%d,%d,%d,%d,%d", &consno, &src, &dst, &type_converter.as_int, &offset);
                type = type_converter.as_ctype;
		PtsToCons cc(type, src, dst);
		if (type == PtsToCons::AddressOf || type == PtsToCons::Copy) {
			addrcopyconstraints.push_back(cc);
		} else if (type == PtsToCons::Load || PtsToCons::Store) {
			loadstoreconstraints.push_back(cc);
		}
        }
        cfile.close();

        return numNodes;
}


void printConstraints(PointsToConstraints &constraints) {
	for (PointsToConstraints::iterator ii = constraints.begin(); ii != constraints.end(); ++ii) {
		ii->print();
	}
}

}

int main(int argc, char** argv) {
  galois::StatManager statManager;
  LonestarStart(argc, argv, name, desc, url);

  numNodes = readConstraints(input.c_str(), addrcopyconstraints, loadstoreconstraints);
  //printConstraints(addrcopyconstraints);
  //printConstraints(loadstoreconstraints);

  result.resize(numNodes);
  nodes.resize(numNodes);
  ocd.init();

  unsigned nodeid = 0;


  for (PointsToInfo::iterator ii = result.begin(); ii != result.end(); ++ii, ++nodeid) {
	ii->init(numNodes);

	GNode src = graph.createNode(Node(nodeid));
	graph.addNode(src);
	nodes[nodeid] = src;
  }

  galois::StatTimer T;
  T.start();

	//numThreads = 0;
  if (numThreads) {
	std::cout << "-------- Parallel version: " << numThreads << " threads.\n";
    runParallel(addrcopyconstraints, loadstoreconstraints);
  } else {
	std::cout << "-------- Sequential version.\n";
    runSerial(addrcopyconstraints, loadstoreconstraints);
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

