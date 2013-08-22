#ifndef GALOISWORKER_H
#define GALOISWORKER_H

#include "GraphGenerator.hxx"
#include "Galois/Graph/LC_Morph_Graph.h"

#include <functional>
#include "TaskDescription.h"

struct ProductionProcess {
public:
	ProductionProcess() { };
	template<typename Context> void operator()(Graph::GraphNode src, Context& ctx);
	std::vector<double> *operator()(TaskDescription &td);
private:
	Graph *graph;
	int atomic_dec(int *value);

};


#endif
