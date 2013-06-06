/* @file
 * @section License
 *Graph which is like LC graphs but it is appendable only
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
 * @author Nikunj Yadav nikunj@cs.utexas.edu
 */

#ifndef LC_MORPH_GRAPH_H_
#define LC_MORPH_GRAPH_H_


#include "Galois/Galois.h"
#include "Galois/LargeArray.h"
#include "Galois/Graph/FileGraph.h"
#include "Galois/Graph/Util.h"
#include "Galois/Runtime/MethodFlags.h"
#include "Galois/gdeque.h"

namespace Galois {
namespace Graph {
#define ES 128
//! Local computation graph (i.e., graph structure does not change)
//! Specialization of LC_Linear_Graph for NUMA architectures
template<typename NodeTy, typename EdgeTy>
class LC_Morph_Graph: private boost::noncopyable {
protected:
	struct NodeInfo;
	typedef GraphImpl::EdgeItem<NodeInfo, EdgeTy, true> EITy;
	//typedef Galois::gdeque<NodeInfo,64> NodeListTy;

	typedef Galois::gdeque<EITy,ES> Edges;
	typedef LargeArray<NodeInfo*> NodesArray;
	typedef typename Edges::iterator eiter;


	struct NodeInfo: public Galois::Runtime::Lockable {
		NodeTy data;
		eiter edgeBegin;
		eiter edgeEnd;
		eiter edgeCurr;
		int numEdges;
		template<typename... Args>
		NodeInfo(Args&& ...args):data(std::forward<Args>(args)...){
		}


	};

	NodesArray nodesArray;


	typedef Galois::InsertBag<NodeInfo> NodeListTy;
	NodeListTy nodes;
	Galois::Runtime::PerThreadStorage<Edges> edges;
	Galois::Runtime::PerThreadStorage<NodeInfo*> last_node;
	uint64_t numNodes;
	uint64_t numEdges;

	struct DistributeInfo {
		uint64_t numNodes;
		uint64_t numEdges;
		FileGraph::iterator begin;
		FileGraph::iterator end;
	};

	//! Divide graph into equal sized chunks
	void distribute(FileGraph& graph, Galois::Runtime::PerThreadStorage<DistributeInfo>& dinfo) {
		size_t total = sizeof(NodeInfo) * numNodes + sizeof(EITy) * numEdges;
		unsigned int num = Galois::getActiveThreads();
		size_t blockSize = total / num;
		size_t curSize = 0;
		FileGraph::iterator ii = graph.begin();
		FileGraph::iterator ei = graph.end();
		FileGraph::iterator last = ii;
		uint64_t nnodes = 0;
		uint64_t nedges = 0;
		uint64_t runningNodes = 0;
		uint64_t runningEdges = 0;

		unsigned int tid;
		for (tid = 0; tid + 1 < num; ++tid) {
			for (; ii != ei; ++ii) {
				if (curSize >= (tid + 1) * blockSize) {
					DistributeInfo& d = *dinfo.getRemote(tid);
					d.numNodes = nnodes;
					d.numEdges = nedges;
					d.begin = last;
					d.end = ii;

					runningNodes += nnodes;
					runningEdges += nedges;
					nnodes = nedges = 0;
					last = ii;
					break;
				}
				size_t nneighbors = std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii));
				nedges += nneighbors;
				nnodes += 1;
				curSize += sizeof(NodeInfo) + sizeof(EITy) * nneighbors;
			}
		}

		DistributeInfo& d = *dinfo.getRemote(tid);
		d.numNodes = numNodes - runningNodes;
		d.numEdges = numEdges - runningEdges;
		d.begin = last;
		d.end = ei;
	}

	struct AllocateNodes {
		Galois::Runtime::PerThreadStorage<DistributeInfo>& dinfo;
		NodeListTy& nodes;
		NodeInfo ** nodesArray;
		FileGraph& graph;
		LC_Morph_Graph *lcGraph;


		AllocateNodes(
				Galois::Runtime::PerThreadStorage<DistributeInfo>& d,
				NodeListTy& n,FileGraph& g,NodeInfo ** nInfo,LC_Morph_Graph *lg):
					dinfo(d), nodes(n), graph(g),nodesArray(nInfo),lcGraph(lg) { }

		void operator()(unsigned int tid, unsigned int num) {

			DistributeInfo& d = *dinfo.getLocal();
			if (!d.numNodes)
				return;

			for (FileGraph::iterator ii = d.begin, ee = d.end; ii != ee; ++ii) {
				NodeInfo *node = lcGraph->createNode(NodeTy());
				lcGraph->addNode(node);
				node->numEdges = std::distance(graph.neighbor_begin(*ii), graph.neighbor_end(*ii));
				nodesArray[*ii]=node;
			}
			*(lcGraph->last_node.getLocal())=nodesArray[*(d.end-1)];
		}
	};

	struct initGraph {
		Galois::Runtime::PerThreadStorage<NodeInfo*> &last_nodes;
		Galois::Runtime::PerThreadStorage<Edges> &local_edges;
		initGraph(Galois::Runtime::PerThreadStorage<NodeInfo*> &l
				,Galois::Runtime::PerThreadStorage<Edges> &e):last_nodes(l),local_edges(e) {
		}
		void operator()(unsigned int tid,unsigned int num) {

			(*local_edges.getLocal()).push_back(EITy(0));
			*last_nodes.getLocal() =NULL;
			return;
			/*NodeInfo *n = new NodeInfo();
			n->edgeEnd=(*local_edges.getLocal()).begin();
			*last_nodes.getLocal()=n;*/
		}

	};


	struct AllocateEdges {
		Galois::Runtime::PerThreadStorage<DistributeInfo>& dinfo;
		Galois::Runtime::PerThreadStorage<Edges>& edges;
		LC_Morph_Graph *lcGraph;
		NodeInfo ** nodesArray;

		FileGraph& graph;

		AllocateEdges(
				Galois::Runtime::PerThreadStorage<DistributeInfo>& d,FileGraph& g,
				Galois::Runtime::PerThreadStorage<Edges>& edges,NodeInfo ** nInfo,LC_Morph_Graph *l):
					dinfo(d),graph(g),nodesArray(nInfo),edges(edges),lcGraph(l) { }
		//! layout the edges
		void operator()(unsigned int tid, unsigned int num) {
			DistributeInfo& d = *dinfo.getLocal();
			if (!d.numNodes)
				return;

			Edges &local_edges = *edges.getLocal();
			int totalEdges=0;
			for (FileGraph::iterator ii = d.begin, ee = d.end; ii != ee; ++ii) {
				int count=0;
				for (FileGraph::neighbor_iterator ni = graph.neighbor_begin(*ii),
						ne = graph.neighbor_end(*ii); ni != ne; ++ni) {
					local_edges.push_back(EITy(nodesArray[*ni],graph.getEdgeData<EdgeTy>(ni)));
					count++;
				}
				nodesArray[*ii]->numEdges=count;
			}
			/*
			 * Hack Alert: You are inserting one additional edge because, you want to keep the iterator to last element of the deque not the end().
			 * If you keep one of the node->edgeEnd iterator (of lets say node called alpha) as edges.end() then any subsequent push backs to the deque will change the end of the deque there by changing the
			 *  edge information of the node (called alpha before).
			 */
			local_edges.push_back(EITy(0));

			eiter it = local_edges.begin();
			for (FileGraph::iterator ii = d.begin, ee = d.end; ii != ee; ii++) {
				nodesArray[*ii]->edgeBegin=it;
				for(int i=0;i<nodesArray[*ii]->numEdges;i++) {
					it++;
				}
				nodesArray[*ii]->edgeEnd=it;
				nodesArray[*ii]->edgeCurr = it;
			}

		}
	};

	public:

	struct makeGraphNode: public std::unary_function<NodeInfo&, NodeInfo*> {
		NodeInfo* operator()(NodeInfo& data) const { return &data; }
	};
	typedef boost::transform_iterator<makeGraphNode,typename NodeListTy::iterator > iterator;
	typedef iterator local_iterator;
	typedef NodeInfo* GraphNode;
	typedef EdgeTy edge_data_type;
	typedef NodeTy node_data_type;
	typedef typename EITy::reference edge_data_reference;
	typedef typename Galois::gdeque<EITy,ES>::iterator edge_iterator;

	~LC_Morph_Graph() {

	}

	NodeTy& getData(const GraphNode& N, MethodFlag mflag = MethodFlag::ALL) {
		Galois::Runtime::checkWrite(mflag, false);
		Galois::Runtime::acquire(N, mflag);
		return N->data;
	}

	edge_data_reference getEdgeData(edge_iterator ni, MethodFlag mflag = MethodFlag::NONE) const {
		/*	Galois::Runtime::checkWrite(mflag, false);
		return ni->get();
		 */
		Galois::Runtime::checkWrite(mflag, false);
		Galois::Runtime::acquire(ni->first(), mflag);
		return *ni->second();
	}

	GraphNode getEdgeDst(edge_iterator ni) const {
		return GraphNode(ni->first());
	}

	uint64_t size() const { return numNodes; }
	uint64_t sizeEdges() const { return numEdges; }

	void setSize(uint64_t numNodes) {
		this->numNodes = numNodes;
	}


	void setEdgeSize(uint64_t numEdges) {
		this->numEdges = numEdges;
	}

	/**
	 * Returns an iterator to all the nodes in the graph. Not thread-safe.
	 */
	iterator begin() {
		return boost::make_transform_iterator(nodes.begin(),makeGraphNode());
	}

	//! Returns the end of the node iterator. Not thread-safe.
	iterator end() {
		return boost::make_transform_iterator(nodes.end(),makeGraphNode());
	}



	local_iterator local_begin() {
		return boost::make_transform_iterator(nodes.local_begin(),makeGraphNode());
	}

	local_iterator local_end() {
		return boost::make_transform_iterator(nodes.local_end(),makeGraphNode());

	}


	edge_iterator edge_begin(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
		Galois::Runtime::acquire(N, mflag);
		if (Galois::Runtime::shouldLock(mflag)) {
			for (edge_iterator ii = N->edgeBegin, ee = N->edgeEnd; ii != ee; ++ii) {
				Galois::Runtime::acquire(ii->first(), mflag);
			}
		}
		return N->edgeBegin;
	}

	edge_iterator edge_end(GraphNode N, MethodFlag mflag = MethodFlag::ALL) {
		return N->edgeEnd;
	}

	template<typename... Args>
	GraphNode createNode(Args&&... args) {
		NodeInfo* N = &(nodes.emplace(std::forward<Args>(args)...));
		Edges &local_edges = *edges.getLocal();
		return GraphNode(N);
	}

	/**
	 * Takes a node which has already been created before and allocates memory for number of specified edges.
	 * last_node is node which was added (with edges), and is stored per thread.
	 */
	void addNode(GraphNode N,int nedges,Galois::MethodFlag mflag = MethodFlag::ALL) {
		Galois::Runtime::checkWrite(mflag, true);
		Galois::Runtime::acquire(N, mflag);
		NodeInfo *l_n = *last_node.getLocal();
		Edges &local_edges = *edges.getLocal();
		N->numEdges=nedges;
		eiter it;
		if(l_n==NULL) {
			it=local_edges.begin();
		}
		else it=l_n->edgeEnd;
		N->edgeBegin=it;
		N->edgeCurr=N->edgeBegin;

		/*Hack Alert:
		 * There is always one additional thing pushed in the edges deque. The edgeBegin of the current node becomes the end (last element of the deque before insertion and not end()) and then
		 * you insert nedges edges to maintain this invariant.
		 */

		for(int i=0;i<nedges;i++) {
			local_edges.push_back(EITy(0));
			it++;
		}
		N->edgeEnd=it;
		*last_node.getLocal()=N;
	}


	template <typename... Args>
	void addNode(const GraphNode& N,Galois::MethodFlag mflag = MethodFlag::ALL) {
		Galois::Runtime::checkWrite(mflag, true);
		Galois::Runtime::acquire(N, mflag);

	}


	edge_iterator addEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = MethodFlag::ALL) {
		Galois::Runtime::checkWrite(mflag, true);
		Galois::Runtime::acquire(src, mflag);
		assert(src->edgeCurr!=src->edgeEnd);

		eiter it = src->edgeBegin;
		while(it!=src->edgeEnd) {
			if(it->first()==dst) {
				return it;
			}
			it++;
		}
		src->edgeCurr->first()=dst;
		it =src->edgeCurr;
		src->edgeCurr++;
		return it;

	}

	edge_iterator addEdgeWithoutCheck(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = MethodFlag::ALL) {
		Galois::Runtime::checkWrite(mflag, true);
		Galois::Runtime::acquire(src, mflag);
		eiter it;
		src->edgeCurr->first()=dst;
		it =src->edgeCurr;
		src->edgeCurr++;
		return it;

	}

	/*
	 * This is a hack. If a thread finishes adding edges of a node in one go, then this works.
	 * No need to allocate the edges before hand, just keep on adding them. This method is not checking whether a node already exists.
	 * That check can be added.
	 */
	edge_iterator addEdgeDynamic(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = MethodFlag::ALL) {
		Galois::Runtime::checkWrite(mflag, true);
		Galois::Runtime::acquire(src, mflag);
		Edges &local_edges = *edges.getLocal();
		NodeInfo *l_n = *last_node.getLocal();
		eiter it;
		if(l_n==NULL) {
			it = local_edges.begin();
		}else it = l_n->edgeEnd;
		if(l_n!=src)
			src->edgeBegin=it;
		it->first()=dst;
		eiter ret = it;
		local_edges.push_back(EITy(0));
		it++;
		src->edgeEnd = it;
		*last_node.getLocal()=src;
		return ret;
	}

	edge_iterator findEdge(GraphNode src, GraphNode dst, Galois::MethodFlag mflag = MethodFlag::ALL) {
		Galois::Runtime::checkWrite(mflag, true);
		Galois::Runtime::acquire(src, mflag);
		eiter it = src->edgeBegin;
		while(it!=src->edgeEnd) {
			if(it->first()==dst) {
				return it;
			}
			it++;
		}
		return src->edgeEnd;
	}


	/*
	 * Test function
	 */
	void checkNode(GraphNode src) {
		int count=std::distance(src->edgeBegin,src->edgeEnd);
		assert(count==src->numEdges);
		Edges &local_edges = *edges.getLocal();
		assert(src->edgeBegin!=local_edges.end());
		assert(src->edgeEnd!=local_edges.end());
		assert(src->edgeCurr==src->edgeEnd);

	}

	/*
	 * Need to be called when an instance of this graph is made. This is to make sure the hack works.
	 * Last Node per thread has to have some deterministic information stored at all times so that we can make the threads
	 * in the following modifications in the graph.
	 */

	void initialize()  {
		Galois::on_each(initGraph(last_node,edges));
	}

	void structureFromFile(const std::string& fname) { Graph::structureFromFile(*this, fname); }

	void structureFromGraph(FileGraph& graph) {
		numNodes = graph.size();
		numEdges = graph.sizeEdges();

		Galois::Runtime::PerThreadStorage<DistributeInfo> dinfo;

		distribute(graph, dinfo);
		nodesArray.create(numNodes);
		unsigned int num = Galois::getActiveThreads();
		Galois::on_each(AllocateNodes(dinfo, nodes, graph,nodesArray.data(),this));
		Galois::on_each(AllocateEdges(dinfo,graph,edges,nodesArray.data(),this));
	}

};

} // end namespace
} // end namespace


#endif /* LC_MORPH_GRAPH_H_ */
