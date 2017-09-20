struct ReconvDrivenWindowing {

	aig::Graph & graph;
	int nInputs;
	int nOutputs;
	int nLevels;

	ReconvDrivenWindowing( aig::Graph & graph, int nInputs, int nOutputs, int nLevels ) 
	: graph( graph ), nInputs( nInputs ), nOutputs( nOutputs ), nLevels( nLevels ) { }

	void operator()( aig::GNode node, galois::UserContext< aig::GNode > & ctx ) const {
		
		aig::NodeData & nodeData = graph.getData( node );
		
		if ( nodeData.type == aig::NodeType::AND ) {
			std::set< aig::GNode > Start, TFO, DagNodes, Window, Leaves, Roots;
			Start.insert( node );
			collectNodesTFO( graph, Start, TFO, nLevels );
			collectRoots( graph, TFO, Roots );
			reconvDrivenCut( graph, Roots, Leaves, Window, DagNodes, nInputs );
			checkRoots( graph, Window, Roots, DagNodes );
			/* 
			Next nodes...
			What is the best way to produce new active nodes 
			that lead to disjoint (non-overlaping) windows?
			*/
		}
	}
};
