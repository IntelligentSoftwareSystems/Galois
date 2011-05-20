/*
 * IndexedGraph.h
 *
 *  Created on: Nov 12, 2010
 *      Author: amshali
 */
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#ifndef INDEXEDGRAPH_H_
#define INDEXEDGRAPH_H_

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/functional.hpp>

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/InsBag.h"
#include "LLVM/SmallVector.h"

namespace Galois {
namespace Graph {

template<typename NodeTy, typename EdgeTy, bool Directional,
		int BranchingFactor = 2>
class IndexedGraph {

	struct gNode: public GaloisRuntime::Lockable {
		//The storage type for edges
		typedef EdgeItem<gNode*, EdgeTy> EITy;
		//The return type for edge data
		typedef typename VoidWrapper<EdgeTy>::ref_type REdgeTy;
		EITy edges[BranchingFactor];
		bool isNullEdges[BranchingFactor];
		NodeTy data;
		bool active;

		gNode(const NodeTy& d, bool a) :
			data(d), active(a) {
			for (int ii = 0; ii < BranchingFactor; ++ii) {
				isNullEdges[ii] = true;
			}
		}

		REdgeTy getEdgeData(gNode* N) {
			for (int ii = 0; ii < BranchingFactor; ++ii)
				if (edges[ii]->getNeighbor() == N)
					return edges[ii]->getData();
			assert(0 && "Edge doesn't exist");
			abort();
		}

		void createEdge(gNode* N, int indx) {
			edges[indx] = EITy(N);
			isNullEdges[indx] = false;
		}

		EITy getEdge(int indx) {
			return edges[indx];
		}

		bool isNullEdge(int indx) {
			return isNullEdges[indx];
		}

		bool isActive() {
			return active;
		}
		~gNode() {
		}
	};

	//The graph manages the lifetimes of the data in the nodes and edges
	typedef GaloisRuntime::galois_insert_bag<gNode> nodeListTy;
	//typedef threadsafe::ts_insert_bag<gNode> nodeListTy;
	nodeListTy nodes;

	//deal with the Node redirction
	inline NodeTy& getData(gNode* ID, MethodFlag mflag = ALL) {
		assert(ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(ID);
		return ID->data;
	}

public:
	class GraphNode {
		friend class IndexedGraph;
		IndexedGraph* Parent;
		gNode* ID;

		explicit GraphNode(IndexedGraph* p, gNode* id) :
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

		inline NodeTy& getData(MethodFlag mflag = ALL) {
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
	// Helpers for the iterator classes
	class makeGraphNode: public std::unary_function<gNode, GraphNode> {
		IndexedGraph* G;
	public:
		makeGraphNode(IndexedGraph* g) :
			G(g) {
		}
		GraphNode operator()(gNode& data) const {
			return GraphNode(G, &data);
		}
	};
	class makeGraphNodePtr: public std::unary_function<gNode*, GraphNode> {
		IndexedGraph* G;
	public:
		makeGraphNodePtr(IndexedGraph* g) :
			G(g) {
		}
		GraphNode operator()(gNode* data) const {
			return GraphNode(G, data);
		}
	};

public:

	// Node Handling

	// Creates a new node holding the indicated data.
	// Node is not added to the graph
	GraphNode createNode(const NodeTy& n) {
		gNode N(n, false);
		return GraphNode(this, &(nodes.push(N)));
	}

	// Adds a node to the graph.
	bool addNode(const GraphNode& n, MethodFlag mflag = ALL) {
		assert(n.ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(n.ID);
		bool oldActive = n.ID->active;
		if (!oldActive) {
			n.ID->active = true;
			//__sync_add_and_fetch(&numActive, 1);
		}
		return !oldActive;
	}

	// Check if a node is in the graph (already added)
	bool containsNode(const GraphNode& n) {
		return n.ID && (n.Parent == this) && n.ID->active;
	}

	// Removes a node from the graph along with all its outgoing/incoming edges.
	// FIXME: incoming edges aren't handled here for directed graphs
	bool removeNode(GraphNode n, MethodFlag mflag = ALL) {
		assert(n.ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(n.ID);
		gNode* N = n.ID;
		bool wasActive = N->active;
		if (wasActive) {
			//__sync_sub_and_fetch(&numActive, 1);
			N->active = false;
			//erase the in-edges first
			for (unsigned int i = 0; i < N->edges.size(); ++i) {
				if (N->edges[i].getNeighbor() != N) // don't handle loops yet
					N->edges[i].getNeighbor()->eraseEdge(N);
			}
			N->edges.clear();
		}
		return wasActive;
	}

	// Edge Handling

	// Adds an edge to the graph containing the specified data.
	void addEdge(GraphNode src, GraphNode dst,
			const typename VoidWrapper<EdgeTy>::type& data, MethodFlag mflag = ALL) {
		assert(src.ID);
		assert(dst.ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(src.ID);
		if (Directional) {
			src.ID->getOrCreateEdge(dst.ID) = data;
		} else {
			if (shouldLock(mflag))
				GaloisRuntime::acquire(dst.ID);
			EdgeTy& E1 = src.ID->getOrCreateEdge(dst.ID);
			EdgeTy& E2 = dst.ID->getOrCreateEdge(src.ID);
			if (src < dst)
				E1 = data;
			else
				E2 = data;
		}
	}

	// Adds an edge to the graph
	void addEdge(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
		assert(src.ID);
		assert(dst.ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(src.ID);
		if (Directional) {
			src.ID->getOrCreateEdge(dst.ID);
		} else {
			if (shouldLock(mflag))
				GaloisRuntime::acquire(dst.ID);
			src.ID->getOrCreateEdge(dst.ID);
			dst.ID->getOrCreateEdge(src.ID);
		}
	}

	void removeEdge(GraphNode src, GraphNode dst, MethodFlag mflag = ALL) {
		assert(src.ID);
		assert(dst.ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(src.ID);
		if (Directional) {
			src.ID->eraseEdge(dst.ID);
		} else {
			if (shouldLock(mflag))
				GaloisRuntime::acquire(dst.ID);
			src.ID->eraseEdge(dst.ID);
			dst.ID->eraseEdge(src.ID);
		}
	}

	typename VoidWrapper<EdgeTy>::type& getEdgeData(GraphNode src, GraphNode dst,
			MethodFlag mflag = ALL) {
		assert(src.ID);
		assert(dst.ID);

		//yes, fault on null (no edge)
		if (shouldLock(mflag))
			GaloisRuntime::acquire(src.ID);

		if (Directional) {
			return src.ID->getEdgeData(dst.ID);
		} else {
			if (shouldLock(mflag))
				GaloisRuntime::acquire(dst.ID);
			if (src < dst)
				return src.ID->getEdgeData(dst.ID);
			else
				return dst.ID->getEdgeData(src.ID);
		}
	}

	// General Things

	int neighborsSize(GraphNode N, MethodFlag mflag = ALL) {
		assert(N.ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(N.ID);
		return N.ID->edges.size();
	}

	//These are not thread safe!!
	typedef boost::transform_iterator<makeGraphNode, boost::filter_iterator<
			std::mem_fun_ref_t<bool, gNode>, typename nodeListTy::iterator> >
			active_iterator;

	active_iterator active_begin() {
		return boost::make_transform_iterator(boost::make_filter_iterator(
				std::mem_fun_ref(&gNode::isActive), nodes.begin(), nodes.end()),
				makeGraphNode(this));
	}

	active_iterator active_end() {
		return boost::make_transform_iterator(boost::make_filter_iterator(
				std::mem_fun_ref(&gNode::isActive), nodes.end(), nodes.end()),
				makeGraphNode(this));
	}
	// The number of nodes in the graph
	unsigned int size() {
		return std::distance(active_begin(), active_end());
	}
	void setNeighbor(GraphNode src, GraphNode dst, int index, MethodFlag mflag =
			ALL) {
		assert(src.ID);

		//yes, fault on null (no edge)
		if (shouldLock(mflag))
			GaloisRuntime::acquire(src.ID);

		src.ID->createEdge(dst.ID, index);
	}
	GraphNode getNeighbor(GraphNode src, int index, MethodFlag mflag = ALL) {
		assert(src.ID);
		if (shouldLock(mflag))
			GaloisRuntime::acquire(src.ID);
		EdgeItem<gNode*, EdgeTy> eity = src.ID->getEdge(index);
		if (!src.ID->isNullEdge(index) && eity.getNeighbor() != NULL)
			return makeGraphNodePtr(this)(eity.getNeighbor()); // FIXME: creating the makeGraphNodePtr every time is not efficient
		else return makeGraphNodePtr(NULL)(NULL);
	}
	IndexedGraph() {
		std::cout << "STAT: NodeSize " << sizeof(gNode) << "\n";
	}
};

}
}

#endif /* INDEXEDGRAPH_H_ */
