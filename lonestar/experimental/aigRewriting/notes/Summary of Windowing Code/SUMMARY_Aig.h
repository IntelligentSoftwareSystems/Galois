#ifndef AIG_AIG_H_
#define AIG_AIG_H_

#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/Graphs/FirstGraph.h"
#include <iostream>
#include <vector>
#include <set>

namespace andInverterGraph {

struct NodeData;

// Nodes hold a NodeData structure, edges hold a boolean value and are directional with InOut distinction
typedef Galois::Graph::FirstGraph< NodeData, bool, true, true > Graph;

typedef Graph::GraphNode GNode;

enum NodeType { AND, PI, PO, LATCH, CONST, CHOICE };

struct NodeData {
	int id;
	NodeType type;
};

class Aig {

private:
	
	Graph graph;
	std::vector< GNode > inputNodes;
	std::vector< GNode > latchNodes;
	std::vector< GNode > outputNodes;
	std::string designName;

public:
	
	Aig();
	virtual ~Aig();

	Graph & getGraph();
	std::vector< GNode > & getInputNodes();
	std::vector< GNode > & getLatchNodes();
	std::vector< GNode > & getOutputNodes();
	std::string getDesignName();
	void setDesignName( std::string designName );

};

} /* namespace aig */

namespace aig = andInverterGraph;

#endif /* AIG_H_ */

