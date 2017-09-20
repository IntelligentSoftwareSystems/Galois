#ifndef AIG_AIG_H_
#define AIG_AIG_H_

#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/FirstGraph.h"
#include <iostream>
#include <vector>
#include <stack>
#include <set>

namespace andInverterGraph {

struct NodeData;

// Nodes hold a NodeData structure, edges hold a boolean value and are directional with InOut distinction
typedef galois::graphs::FirstGraph< NodeData, bool, true, true > Graph;

typedef Graph::GraphNode GNode;

enum NodeType { AND, PI, PO, LATCH, CONST, CHOICE };

struct NodeData {
	int id;
	NodeType type;
	int counter;
	bool andBehavior;
	//std::string name;
	//std::vector< std::set< GNode > > cuts;
	//std::set< GNode > bestCut;
	NodeData( ) : counter( 0 ), andBehavior( true ) { }
};

class Aig {

private:
	
	Graph graph;
	std::string designName;
	std::vector< GNode > inputNodes;
	std::vector< GNode > latchNodes;
	std::vector< GNode > outputNodes;
	std::vector< std::string > inputNames;
	std::vector< std::string > latchNames;
	std::vector< std::string > outputNames;

	void topologicalSort( GNode node, std::vector< bool > & visited, std::stack< GNode > & stack );

public:
	
	Aig();
	virtual ~Aig();

	Graph & getGraph();
	std::vector< GNode > & getInputNodes();
	std::vector< GNode > & getLatchNodes();
	std::vector< GNode > & getOutputNodes();
	std::vector< std::string > & getInputNames();	
	void setInputNames( std::vector< std::string > inputNames );
	std::vector< std::string > & getLatchNames();
	void setLatchNames( std::vector< std::string > latchNames );
	std::vector< std::string > & getOutputNames();
	void setOutputNames( std::vector< std::string > outputNames );
	
	int getNumInputs();
	int getNumLatches();
	int getNumOutputs();
	int getNumAnds();

	std::string getDesignName();
	void setDesignName( std::string designName );
	
	void resetAndIDs();
	void computeTopologicalSort( std::stack< GNode > & stack );
	void writeDot( std::string path, std::string dotText );
	std::string toDot();


};

} /* namespace aig */

namespace aig = andInverterGraph;

#endif /* AIG_H_ */

