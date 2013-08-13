#include "GaloisWorker.h"
#include "Point2D/MatrixGenerator.hxx"
#include <vector>

class TestFunction : public IDoubleArgFunction {
	double ComputeValue(double x, double y) {
		return 1.0;
	}
};

template<typename Context>
void ProductionProcess::operator()(Graph::GraphNode src, Context& ctx)
{
	Node node = src->data;
	switch (node.productionToExecute) {
	case A1:
		node.productions->A1(node.v, node.input);
		break;
	case A:
		node.productions->A(node.v, node.input);
		break;
	case AN:
		node.productions->AN(node.v, node.input);
		break;
	case A2:
		node.productions->A2(node.v);
		break;
	case E:
		node.productions->E(node.v);
		break;
	case EROOT:
		node.productions->ERoot(node.v);
		break;
	case BS:
		node.productions->BS(node.v);
		break;
	default:
		break;
	}

	std::cout << "jestem w wierzcholku " << src->data.x << "\n";
	for(LCM_edge_iterator ii = src->edgeBegin, ei = src->edgeEnd; ii != ei; ++ii)
	{
		std::cout << "przechodze przez krawedz\n";
		GraphNode graphNode = graph->getEdgeDst(ii,Galois::MethodFlag::NONE);

		std::cout << "mam polaczenie z " << graphNode->data.x << "\n";

		int nr_of_incoming_edges = atomic_dec(&graphNode->data.nr_of_incoming_edges);

		if(!nr_of_incoming_edges)
			ctx.push(graphNode);
	}

}

void ProductionProcess::operator()()
{
	// implement everything is needed to input data to solver,
	// preprocessing,
	IDoubleArgFunction *function = new TestFunction();
	GraphGenerator* generator = new GraphGenerator();
	AbstractProduction *production = new AbstractProduction(5, 17, 21, 21);

	MatrixGenerator *matrixGenerator = new MatrixGenerator();
	std::list<Tier*> *tiers = matrixGenerator->CreateMatrixAndRhs(6, 0, 0, 1, function);
	Mes2DPreprocessor *preprocessor = new Mes2DPreprocessor();
	std::vector<EquationSystem *> *inputMatrices = preprocessor->preprocess(tiers);
	generator->generateGraph(6, production, inputMatrices);

	graph = generator->getGraph();
	LCM_iterator it = graph->begin();
	std::vector<GraphNode> initial_nodes_vector;
	while(it != graph->end())
	{
		GraphNode graphNode = *(it);
		if(graphNode->data.nr_of_incoming_edges == 0)
			initial_nodes_vector.push_back(graphNode);
		++it;
	}
	std::vector<GraphNode>::iterator iii = initial_nodes_vector.begin();
	while(iii != initial_nodes_vector.end()){
		Galois::for_each(*iii,*this);
		++iii;
	}

}

int ProductionProcess::atomic_dec(int *value) {
	// XXX: more portable solution?
	return __sync_add_and_fetch(value, -1);
}
