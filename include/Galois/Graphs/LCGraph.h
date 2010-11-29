// simple graph -*- C++ -*-

#include <list>
#include <map>
#include <vector>
#include <iostream>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/functional.hpp>

#include "Galois/Runtime/Context.h"

#include "Galois/Graphs/Graph.h"


namespace Galois {
namespace Graph {

template<typename NodeTy, typename EdgeTy, bool Directional>
class LCGraph {
	std::vector<EdgeTy> edgeData;
	std::vector<int> outIdx;
	std::vector<int> outs;

	int numNodes;

	struct gNode: public GaloisRuntime::Lockable {

		NodeTy data;
		bool active;
		int idx;

		gNode(const NodeTy& d, bool a, int idx) :
			data(d), active(a), idx(idx) {
		}

		bool isActive() {
			return active;
		}
	};

	std::vector<gNode> nodes;



public:
	class GraphNode {
		friend class LCGraph;
		LCGraph* Parent;
		gNode* ID;

		explicit GraphNode(LCGraph* p, gNode* id) :
			Parent(p), ID(id) {
		}

	public:

		GraphNode() :
			Parent(0), ID(0) {
		}

		NodeTy& getData(MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
			return Parent->getData(ID, mflag, C);
		}

		bool isNull() const {
			return !Parent;
		}

		bool operator!=(const GraphNode& rhs) const {
			return Parent != rhs.Parent || ID != rhs.ID;
		}

		bool operator==(const GraphNode& rhs) const {
			return Parent == rhs.Parent && ID == rhs.ID;
		}

		bool operator<(const GraphNode& rhs) const {
			return Parent < rhs.Parent || (Parent == rhs.Parent && ID < rhs.ID);
		}

		bool operator>(const GraphNode& rhs) const {
			return Parent > rhs.Parent || (Parent == rhs.Parent && ID > rhs.ID);
		}

	};


private:
	//deal with the Node redirection
	template<typename Context>
	NodeTy& getData(gNode* ID, MethodFlag mflag = ALL, Context* C = getThreadContext()) {
		assert(ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(ID);
		return ID->data;
	}

	int getEdgeIdx(GraphNode src, GraphNode dst){
		int idx = getId(src);
		int target = getId(dst);
		int start = outIdx.at(idx);
		int end = outIdx.at(idx+1);

		for(int i = start; i < end; i++){
			if(outs[i]==target){
				return i;
			}
		}
		std::cout<<"Error getting Edge idx. Exiting.";
		exit(0);
	}

	int getId(GraphNode N){
		return N.ID->idx;
	}

	int neighborsSize(GraphNode N, int adjIdx[], int adj[]) {

		int idx = getId(N);
		int start = adjIdx[idx];
		int end = adjIdx[idx+1];

		return end - start;
	}

	// Helpers for the iterator classes
	class makeGraphNode: public std::unary_function<gNode, GraphNode> {
		LCGraph* G;
	public:
		makeGraphNode(LCGraph* g) :
			G(g) {
		}
		GraphNode operator()(gNode& data) const {
			return GraphNode(G, &data);
		}
	};
	class makeGraphNodePtr: public std::unary_function<gNode*, GraphNode> {
		LCGraph* G;
	public:
		makeGraphNodePtr(LCGraph* g) :
			G(g) {
		}
		GraphNode operator()(gNode* data) const {
			return GraphNode(G, data);
		}
	};

	class makeGraphNode3 : public std::unary_function<int, GraphNode>{
		LCGraph* G;
	public:
		makeGraphNode3(LCGraph* g) : G(g) {}
		GraphNode operator()(int idx) const {
		return GraphNode(G, &(G->nodes.at(idx)));
		}
	};


public:
	// Create Graph
	typedef FirstGraph<NodeTy,EdgeTy,Directional> FGraph;

	void createGraph(FGraph *g){

		typedef typename FGraph::GraphNode GNode;
		typedef typename FGraph::active_iterator GraphActiveIterator;
		typedef typename FGraph::neighbor_iterator GraphNeighborIterator;
		typedef std::pair<GNode, int> GIPair;

		const unsigned int numNodes = g->size();

		std::vector<GNode> rnodes(numNodes);
		std::map<GNode, int> nodeMap;

		int idx = 0;
		for (GraphActiveIterator ii = g->active_begin(), ee = g->active_end();
				ii != ee; ++ii){
			idx = nodeMap.size();
			nodeMap.insert(GIPair(*ii,idx));
			nodes.push_back(gNode(ii->getData(), true, idx));
			rnodes.at(idx) = *ii;

		}

		outIdx.push_back(0);
		for(unsigned int i = 0; i < numNodes; i++){
			const GNode src = rnodes[i];

			for(GraphNeighborIterator ii = g->neighbor_begin(src), ee = g->neighbor_end(src);
					ii != ee; ++ii){
				outs.push_back(nodeMap.find(*ii)->second);
				EdgeTy et = g->getEdgeData(src, *ii);
				edgeData.push_back(et);
			}


			outIdx.push_back(outs.size());
		}

		this->numNodes = numNodes;
	}

	// Check if a node is in the graph (already added)
	bool containsNode(GraphNode n) {
		int idx = getId(n);
		return 0 <= idx && idx <= numNodes;
	}

	// Edge Handling
	typename VoidWrapper<EdgeTy>::type& getEdgeData(const GraphNode& src,
			const GraphNode& dst, MethodFlag mflag = ALL,
			SimpleRuntimeContext* C = getThreadContext()) {
		assert(src.ID);
		assert(dst.ID);

	    if (shouldLock(mflag))
	      SimpleRuntimeContext::acquire(C, src.ID);

		return edgeData[getEdgeIdx(src, dst)];
	}

	// General Things
    typedef boost::transform_iterator<makeGraphNode3,
  			  	  typename std::vector<int>::iterator> neighbor_iterator;

	neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
	    assert(N.ID);
	    if (shouldLock(mflag))
	      SimpleRuntimeContext::acquire(C, N.ID);
		typename std::vector<int>::iterator I = outs.begin() + outIdx[N.ID->idx];
		return boost::make_transform_iterator(I,
				makeGraphNode3(this));
	}
	neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
	    assert(N.ID);
	    if (shouldLock(mflag)) // Probably not necessary (no valid use for an end pointer should ever require it)
	      SimpleRuntimeContext::acquire(C, N.ID);
		typename std::vector<int>::iterator I = outs.begin() + outIdx[(N.ID->idx)+1];
		return boost::make_transform_iterator(I,
				makeGraphNode3(this));
	}

	//These are not thread safe!!
	typedef boost::transform_iterator<LCGraph::makeGraphNode, boost::filter_iterator<std::mem_fun_ref_t<bool, gNode>, typename std::vector<gNode>::iterator> >
			active_iterator;

	active_iterator active_begin() {
	return boost::make_transform_iterator(boost::make_filter_iterator(
									  std::mem_fun_ref(&gNode::isActive), nodes.begin(), nodes.end()),
					  LCGraph::makeGraphNode(this));
	}

	active_iterator active_end() {
	return boost::make_transform_iterator(boost::make_filter_iterator(
									  std::mem_fun_ref(&gNode::isActive), nodes.end(), nodes.end()),
					  LCGraph::makeGraphNode(this));
	}

	int neighborsSize(GraphNode N, MethodFlag mflag = ALL, SimpleRuntimeContext* C = getThreadContext()) {
		assert(N.ID);
		if (shouldLock(mflag))
		  SimpleRuntimeContext::acquire(C, N.ID);
		return neighborsSize(N, outIdx, outs);
	}
	// The number of nodes in the graph
	unsigned int size() {
		return std::distance(active_begin(), active_end());
	}

	LCGraph() {
		std::cout << "STAT: NodeSize " << (int) sizeof(gNode) << "\n";
	}
};

}
}
