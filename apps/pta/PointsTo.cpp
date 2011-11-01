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
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/util/SparseBitVector.h"
#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"
#include "Galois/Runtime/WorkList.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <stack>

namespace {

const char* name = "Points-to Analysis";
const char* description = "Performs inclusion-based points-to analysis over the input constraints.";
const char* url = NULL;
const char* help = "<constraints>";

const unsigned THRESHOLD_LOADSTORE = 1000;
const unsigned THRESHOLD_OCD = 1000;

typedef Galois::Graph::FirstGraph<unsigned,void,true> Graph;	// first arg is node id.
typedef Graph::GraphNode GNode;
typedef std::pair<GNode,GNode> Edge;
typedef std::vector<GNode> WorkList;

class PtsToCons {
public:
	typedef enum {AddressOf, Copy, Load, Store} ConstraintType;

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
typedef std::vector<Galois::SparseBitVector> PointsToInfo;

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
	for (WorkList::iterator ii = worklist.begin(); ii != worklist.end(); ++ii) {
		unsigned nodeid = graph.getData(*ii);
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
	for (Graph::neighbor_iterator ii = graph.neighbor_begin(nn), ei = graph.neighbor_end(nn); ii != ei; ++ii) {
		unsigned iiid = graph.getData(*ii);
		if (cycleDetect(iiid, cyclenode)) {
			return true;	// don't pop from ancestors.
		}
	}
	ancestors.pop_back();
	return false;
  }
  void cycleCollapse(unsigned repr) {
	// assert(repr is present in ancestors).
	static unsigned ncycles = 0;
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
	ancestors.clear();	// since collapse is called from top level process(), the ancestors need to be cleared for the next element in the worklist.
  }
  void makeRepr(unsigned nodeid, unsigned repr) {
	// make repr the representative of nodeid.
	if (repr != nodeid) {
		//std::cout << "debug: repr[" << nodeid << "] = " << repr << std::endl;
		representative[nodeid] = repr;
		result[repr].unify(result[nodeid]);
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
		result[repr].print(prefix);
	}
}
bool propagate(GNode &src, GNode &dst) {
	unsigned srcid = graph.getData(src);
	unsigned dstid = graph.getData(dst);

	if (srcid != dstid) {
		unsigned srcreprid = ocd.getFinalRepresentative(srcid);
		unsigned dstreprid = ocd.getFinalRepresentative(dstid);
		if (srcreprid != dstreprid) {
			//std::cout << "debug: unifying " << dstreprid << " by " << srcreprid << std::endl;
			return result[dstreprid].unify(result[srcreprid]);
		}
	}
	return false;	// no change in points-to info.
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
			graph.addEdge(nodes[src], nodes[dst]);
			//std::cout << "debug: adding edge from " << src << " to " << dst << std::endl;
			worklist.push_back(nodes[src]);
		}
	}
}
void processLoadStore(PointsToConstraints &constraints, WorkList &worklist) {
	// add edges to the graph based on points-to information of the nodes
	// and then add the source of each edge to the worklist.
	for (PointsToConstraints::iterator ii = constraints.begin(); ii != constraints.end(); ++ii) {
		unsigned src, dst;
		//std::cout << "debug: Processing constraint: "; ii->print();
		ii->getSrcDst(src, dst);
		unsigned srcrepr = ocd.getFinalRepresentative(src);
		unsigned dstrepr = ocd.getFinalRepresentative(dst);
		if (ii->getType() == PtsToCons::Load) {
			std::vector<unsigned> ptstoOfSrc;
			result[srcrepr].getAllSetBits(ptstoOfSrc);
			for (std::vector<unsigned>::iterator pointee = ptstoOfSrc.begin(); pointee != ptstoOfSrc.end(); ++pointee) {
				unsigned pointeerepr = ocd.getFinalRepresentative(*pointee);
				if (pointeerepr != dstrepr && !nodes[pointeerepr].hasNeighbor(nodes[dstrepr])) {
					graph.addEdge(nodes[pointeerepr], nodes[dstrepr]);
					//std::cout << "debug: adding edge from " << *pointee << " to " << dst << std::endl;
					worklist.push_back(nodes[pointeerepr]);
				}
			}
		} else {	// store.
			std::vector<unsigned> ptstoOfDst;
			bool newedgeadded = false;
			result[dstrepr].getAllSetBits(ptstoOfDst);
			for (std::vector<unsigned>::iterator pointee = ptstoOfDst.begin(); pointee != ptstoOfDst.end(); ++pointee) {
				unsigned pointeerepr = ocd.getFinalRepresentative(*pointee);
				if (srcrepr != pointeerepr && !nodes[srcrepr].hasNeighbor(nodes[pointeerepr])) {
					graph.addEdge(nodes[srcrepr], nodes[pointeerepr]);
					//std::cout << "debug: adding edge from " << src << " to " << *pointee << std::endl;
					newedgeadded = true;
				}
			}
			if (newedgeadded) {
				worklist.push_back(nodes[srcrepr]);
			}
		}
	}
}

struct Process {
  Process() { }

  template<typename Context>
  void operator()(GNode src, Context& ctx) {
    for (Graph::neighbor_iterator ii = graph.neighbor_begin(src),
        ei = graph.neighbor_end(src); ii != ei; ++ii) {
      	GNode dst = *ii;
	if (propagate(src, dst)) {
		ctx.push(dst);
	}
    }
  }
};

void runSerial(PointsToConstraints &addrcopyconstraints, PointsToConstraints &loadstoreconstraints) {
	WorkList worklist;
	unsigned niteration = 0;
	bool changed = false;
	unsigned nnodesprocessed = 0;

	processAddressOfCopy(addrcopyconstraints, worklist);
        processLoadStore(loadstoreconstraints, worklist);	// required when there are zero copy constraints which keeps worklist empty.

	//std::cout << "debug: no of addr+copy constraints = " << addrcopyconstraints.size() << ", no of load+store constraints = " << loadstoreconstraints.size() << std::endl;
	//std::cout << "debug: no of nodes = " << nodes.size() << std::endl;

	while (!worklist.empty()) {
  		//std::cout << "debug: Iteration " << ++niteration << ", worklist.size=" << worklist.size() << "\n"; 
		GNode src = worklist.back();
		worklist.pop_back();

		//std::cout << "debug: processing worklist element " << graph.getData(src) << std::endl;
    		for (Graph::neighbor_iterator ii = graph.neighbor_begin(src), ei = graph.neighbor_end(src); ii != ei; ++ii) {
      			GNode dst = *ii;
			if (propagate(src, dst)) {
				worklist.push_back(dst);
			}
		}
		if (++nnodesprocessed > THRESHOLD_LOADSTORE || worklist.empty()) {
			nnodesprocessed = 0;
	        	processLoadStore(loadstoreconstraints, worklist);	// add edges to graph, add their sources to worklist.
			if (worklist.size() > THRESHOLD_OCD) {
				ocd.process(worklist);
			}
		}
        }
}

void runParallel(PointsToConstraints &addrcopyconstraints, PointsToConstraints &loadstoreconstraints) {
	WorkList worklist;
	unsigned niteration = 0;
	
	processAddressOfCopy(addrcopyconstraints, worklist);
	processLoadStore(loadstoreconstraints, worklist);

	while (!worklist.empty()) {
		//std::cout << "debug: iteration " << ++niteration << std::endl;
		using namespace GaloisRuntime::WorkList;
	  	Galois::for_each<LocalQueues<dChunkedFIFO<1024> > >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<FIFO<> >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<ChunkedFIFO<1024> >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<dChunkedFIFO<1024> >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<ChunkedLIFO<32> >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<LocalQueues<ChunkedLIFO<32> > >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<LocalQueues<FIFO<1024> > >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<LocalQueues<ChunkedFIFO<32> > >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<LocalQueues<dChunkedLIFO<32>, FIFO<> > >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<LocalQueues<ChunkedLIFO<1024>, FIFO<> > >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each<LocalQueues<ChunkedFIFO<1024>, FIFO<> > >(worklist.begin(), worklist.end(), Process());
	  	//Galois::for_each(worklist.begin(), worklist.end(), Process());
		worklist.clear();	// important.
		processLoadStore(loadstoreconstraints, worklist);
		ocd.process(worklist);	// changes the graph, doesn't change the worklist.
	}
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
		sscanf(cstr.c_str(), "%d,%d,%d,%d,%d", &consno, &src, &dst, (int *)&type, &offset);
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

int main(int argc, const char** argv) {
  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() != 1) {
    std::cerr 
      << "incorrect number of arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  numNodes = readConstraints(args[0], addrcopyconstraints, loadstoreconstraints);
  //printConstraints(addrcopyconstraints);
  //printConstraints(loadstoreconstraints);

  result.resize(numNodes);
  nodes.resize(numNodes);
  ocd.init();

  unsigned nodeid = 0;


  for (PointsToInfo::iterator ii = result.begin(); ii != result.end(); ++ii, ++nodeid) {
	ii->init(numNodes);

	GNode src = graph.createNode(nodeid);
	graph.addNode(src);
	nodes[nodeid] = src;
  }
  //std::cout << "constraint reading and initialization over.\n";

  Galois::StatTimer T;
  T.start();

	//numThreads = 0;
  if (numThreads) {
	//std::cout << "-------- Parallel version: " << numThreads << " threads.\n";
    runParallel(addrcopyconstraints, loadstoreconstraints);
  } else {
	//std::cout << "-------- Sequential version.\n";
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

