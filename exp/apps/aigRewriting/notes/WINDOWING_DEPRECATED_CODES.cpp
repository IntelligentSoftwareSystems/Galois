void reconvDrivenBackCut( aig::Graph & graph, GNodeSet & sources, GNodeSet & roots, GNodeSet & window, GNodeSet & dagNodes, int cutSizeLimit, Galois::PerIterAllocTy & allocator ) const {

		GNodeSet currentRoots( allocator );	
		//GNodeSet currentRoots( nBuckets, hasher, comparator, allocator );

		for ( aig::GNode node : sources ) {
			// window.clear();
			currentRoots.clear();
			window.insert( node );
			currentRoots.insert( node );
			constructBackCut( graph, currentRoots, window, dagNodes, cutSizeLimit );
			unification( roots, currentRoots );
		}
	}

	void constructBackCut( aig::Graph & graph, GNodeSet & roots, GNodeSet & window, GNodeSet & dagNodes, int cutSizeLimit ) const {
	
		aig::GNode minCostNode;
		int minCost = std::numeric_limits<int>::max();
		bool onlyPOs = true;
		for ( aig::GNode node : roots ) {
			aig::NodeData & nodeData = graph.getData( node, Galois::MethodFlag::READ );
			if ( nodeData.type != aig::NodeType::PO ) {
				int cost = rootCost( graph, node, window );
				if ( minCost > cost ) {
					minCost = cost;
					minCostNode = node;
					onlyPOs = false;
				}
			}
		}
	
		if ( onlyPOs || ( (roots.size()+minCost) > cutSizeLimit ) ) {
			return;
		}

		roots.erase( minCostNode );
		for ( auto edge : graph.out_edges( minCostNode ) ) {
			aig::GNode currentNode = graph.getEdgeDst( edge );
			roots.insert( currentNode );
			window.insert( currentNode );

			int fanout = std::distance( graph.out_edges( currentNode ).begin(), graph.out_edges( currentNode ).end() );
			aig::NodeData & currentNodeData = graph.getData( currentNode, Galois::MethodFlag::READ );

			if ( fanout > 1 && (currentNodeData.type == aig::NodeType::AND) ) { // DagNodes are stored in a separated list taht is used to discover side outputs of window
				dagNodes.insert( currentNode );
			}
		}

		constructBackCut( graph, roots, window, dagNodes, cutSizeLimit );
	}

	int rootCost( aig::Graph & graph, aig::GNode & node, GNodeSet & window ) const {

		int cost = -1;
		for ( auto edge : graph.out_edges( node ) ) {
			aig::GNode currentNode = graph.getEdgeDst( edge );
			auto it = window.find( currentNode );
			if ( it == window.end() ) {
				cost++;
			}
		}
		return cost;
	}
