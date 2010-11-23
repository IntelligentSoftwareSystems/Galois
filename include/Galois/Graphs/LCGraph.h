// simple graph -*- C++ -*-

#include <list>
#include <map>
#include <vector>
#include <iostream>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/functional.hpp>

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/InsBag.h"
#include "Support/ThreadSafe/TSIBag.h"
#include "LLVM/SmallVector.h"

#include "Galois/Graphs/Graph.h"


namespace Galois {
namespace Graph {

template<typename NodeTy, typename EdgeTy, bool Directional>
class LCGraph {
	typedef FirstGraph<NodeTy,EdgeTy,Directional> FGraph;
	typedef typename FGraph::GraphNode GNode;

	std::vector<EdgeTy> edgeData;
	std::vector<int> inIdx;
	std::vector<int> ins;
	std::vector<int> outIdx;
	std::vector<int> outs;

	int numNodes;

	struct gNode: public GaloisRuntime::Lockable {
		//The storage type for edges
		typedef EdgeItem<gNode*, EdgeTy> EITy;
		//The return type for edge data
		typedef typename VoidWrapper<EdgeTy>::ref_type REdgeTy;
		typedef llvm::SmallVector<EITy, 3> edgesTy;
		edgesTy edges;
		NodeTy data;
		bool active;

		typedef typename edgesTy::iterator iterator;

		iterator begin() {
			return edges.begin();
		}
		iterator end() {
			return edges.end();
		}

		typedef typename boost::transform_iterator<boost::mem_fun_ref_t<gNode*,
				EITy>, iterator> neighbor_iterator;

		neighbor_iterator neighbor_begin() {
			return boost::make_transform_iterator(begin(), boost::mem_fun_ref(
					&EITy::getNeighbor));
		}
		neighbor_iterator neighbor_end() {
			return boost::make_transform_iterator(end(), boost::mem_fun_ref(
					&EITy::getNeighbor));
		}

		gNode(const NodeTy& d, bool a) :
			data(d), active(a) {
		}

		void prefetch_neighbors() {
			for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
				if (ii->getNeighbor())
					__builtin_prefetch(ii->getNeighbor());
		}

		void eraseEdge(gNode* N) {
			for (iterator ii = begin(), ee = end(); ii != ee; ++ii) {
				if (ii->getNeighbor() == N) {
					edges.erase(ii);
					return;
				}
			}
		}

		REdgeTy getEdgeData(gNode* N) {
			for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
				if (ii->getNeighbor() == N)
					return ii->getData();
			assert(0 && "Edge doesn't exist");
			abort();
		}

		REdgeTy getOrCreateEdge(gNode* N) {
			for (iterator ii = begin(), ee = end(); ii != ee; ++ii)
				if (ii->getNeighbor() == N)
					return ii->getData();
			edges.push_back(EITy(N));
			return edges.back().getData();
		}

		bool isActive() {
			return active;
		}
	};

	std::vector<gNode> nodes;


		/*EdgeTy *edgeData;
		int *inIdx;
		int *ins;
		int *outIdx;
		int *outs;
		*/

	//The graph manages the lifetimes of the data in the nodes and edges
	//typedef GaloisRuntime::galois_insert_bag<gNode> nodeListTy;
	//typedef threadsafe::ts_insert_bag<gNode> nodeListTy;
	//nodeListTy nodes;

	//deal with the Node redirction
	NodeTy& getData(gNode* ID, MethodFlag mflag = ALL) {
		assert(ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(ID);
		return ID->data;
	}



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

		void prefetch_all() {
			if (ID)
				ID->prefetch_neighbors();
		}

		NodeTy& getData(MethodFlag mflag = ALL) {
			return Parent->getData(ID, mflag);
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


	int getEdgeIdx(GraphNode src, GraphNode dst){
		int idx = getId(src);
		int target = getId(dst);
		int &start = outIdx.at(idx);//outIdx[idx];
		int &end = outIdx.at(idx+1);

		for(int i = start; i < end; i++){
			if(outs[i]==target){
				return i;
			}
		}
		return -1;
	}

	int getId(GraphNode N){
		return N.ID->data.id;
	}

	int neighborsSize(GraphNode N, int adjIdx[], int adj[]) {
		assert(N.ID);
		/*if (shouldLock(mflag))
			GaloisRuntime::acquire(N.ID);
		return N.ID->edges.size();*/
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

public:

	// Creates a new node holding the indicated data.
	  // Node is not added to the graph
	GraphNode createNode(const NodeTy& n) {
		gNode N(n, false);
		nodes.push_back(N);
		return GraphNode(this, &(nodes.back()));
	}
	// Create Graph
	typedef typename FGraph::active_iterator GraphActiveIterator;
	typedef typename FGraph::neighbor_iterator GraphNeighborIterator;

	void createGraph(FGraph *g){


		const unsigned int numNodes = g->size();
		//this->nodes = std::vector<gNode>(numNodes);
		//this->nodes.resize(10);

		std::vector<GNode> rnodes(numNodes);
		std::map<GNode, int> nodeMap;

		typedef std::pair<GNode, int> GIPair;

		//int idx = 0;
		for (GraphActiveIterator ii = g->active_begin(), ee = g->active_end();
				ii != ee; ++ii){
			int idx = nodeMap.size();
			nodeMap.insert(GIPair(*ii,idx));
			//nodes.insert(nodes.begin()+idx, gNode(ii->getData(), true));
			nodes.push_back(gNode(ii->getData(), true));
			//gNode N(ii->getData(), true);
			//nodes.push_back(N);
			//rnodes.insert(rnodes.begin()+idx,*ii);
			rnodes.at(idx) = *ii;

		}

		/*std::vector<int> ins;
		std::vector<int> inIdx;
		std::vector<int> outs;
		std::vector<int> outIdx;*/
		//std::list<EdgeTy> edgeData;


		inIdx.push_back(0);
		outIdx.push_back(0);
		for(unsigned int i = 0; i < numNodes; i++){
			const GNode src = rnodes[i];
			for (GraphNeighborIterator ii = g->neighbor_begin(src), ee = g->neighbor_end(src);
					ii != ee; ++ii){
				ins.push_back(nodeMap.find(*ii)->second);
			}
			inIdx.push_back(ins.size());

			for(GraphNeighborIterator ii = g->neighbor_begin(src), ee = g->neighbor_end(src);
					ii != ee; ++ii){
				outs.push_back(nodeMap.find(*ii)->second);
				edgeData.push_back(g->getEdgeData(src, *ii));
			}
			outIdx.push_back(outs.size());
		}

		/*this->inIdx.assign(inIdx.begin(), inIdx.end());
		this->ins.assign(ins.begin(), ins.end());
		this->outIdx.assign(outIdx.begin(), outIdx.end());
		this->outs.assign(outs.begin(), outs.end());

		std::vector<int>::iterator p;
		int i = 0;

		p=inIdx.begin();
		while(p!=inIdx.end()){
			this->inIdx.push_back(*p);//[i++]=*p;
		}

		p = ins.begin(); i = 0;
		while(p!=ins.end()){
			this->ins.push_back(*p);//insert(this->ins.begin()+i++, *p);//[i++]=*p;
		}

		p = outIdx.begin(); i = 0;
		while(p!=outIdx.end()){
			this->outIdx.push_back(*p);//insert(this->outIdx.begin()+i++,*p);//[i++]=*p;
		}

		p = outs.begin(); i = 0;
		while(p!=outs.end()){
			this->outs.push_back(*p);//insert(this->outs.begin()+i++,*p);//[i++]=*p;
		}
*/
		/*typedef typename std::list<EdgeTy>::iterator EdgeTyListIterator;
		EdgeTyListIterator pl = edgeData.begin(); //i = 0;
		while(pl!=edgeData.end()){
			this->edgeData.push_back(*pl);//.insert(this->edgeData.begin()+i++,*pl);//[i++] = *pl;
		}*/

		this->numNodes = numNodes;
	}
	// Node Handling



	// Check if a node is in the graph (already added)
	bool containsNode(GraphNode n) {
		int idx = getId(n);
		return 0 <= idx && idx <= numNodes;
		//return n.ID && (n.Parent == this) && n.ID->active;
	}

	// Edge Handling


	typename VoidWrapper<EdgeTy>::type& getEdgeData(const GraphNode& src,
			const GraphNode& dst, MethodFlag mflag = ALL) {
		assert(src.ID);
		assert(dst.ID);

		//yes, fault on null (no edge)
		if (shouldLock(mflag))
			GaloisRuntime::acquire(src.ID);

		/*if (Directional) {
			return src.ID->getEdgeData(dst.ID);
		} else {
			if (shouldLock(mflag))
				GaloisRuntime::acquire(dst.ID);
			if (src < dst)
				return src.ID->getEdgeData(dst.ID);
			else
				return dst.ID->getEdgeData(src.ID);
		}*/
		return edgeData.at(getEdgeIdx(src, dst));
		//return retval;
	}

	// General Things



	typedef typename boost::transform_iterator<makeGraphNodePtr,
			typename gNode::neighbor_iterator> neighbor_iterator;

	neighbor_iterator neighbor_begin(GraphNode N, MethodFlag mflag = ALL) {
		assert(N.ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(N.ID);
		for (typename gNode::neighbor_iterator ii = N.ID->neighbor_begin(), ee =
				N.ID->neighbor_end(); ii != ee; ++ii) {
			__builtin_prefetch(*ii);
			if (!Directional && shouldLock(mflag))
				GaloisRuntime::acquire(*ii);
		}
		return boost::make_transform_iterator(N.ID->neighbor_begin(),
				makeGraphNodePtr(this));
	}
	neighbor_iterator neighbor_end(GraphNode N, MethodFlag mflag = ALL) {
		assert(N.ID);
		if (shouldLock(mflag)) // Probably not necessary (no valid use for an end pointer should ever require it)
			GaloisRuntime::acquire(N.ID);
		return boost::make_transform_iterator(N.ID->neighbor_end(),
				makeGraphNodePtr(this));
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

	/*int outNeighborSize(GraphNode N, MethodFlag mflag = ALL){
		// TODO
	}

	int inNeighborSize(GraphNode N, MethodFlag mflag = ALL){
		// TODO
	}*/
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
