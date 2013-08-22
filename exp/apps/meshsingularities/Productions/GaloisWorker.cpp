#include <vector>

#include "GaloisWorker.h"
#include "Processing.h"

#include "Galois/Statistic.h"

#include "Point3D/MatrixGenerator.hxx"
#include "Point3D/TripleArgFunction.hxx"


/*
class TestFunction : public IDoubleArgFunction {
	double ComputeValue(double x, double y) {
		return 1.0;
	}
}; */

using namespace D3;

class TestFunction3D : public ITripleArgFunction {
	double ComputeValue(double x, double y, double z) {
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

	for(LCM_edge_iterator ii = src->edgeBegin, ei = src->edgeEnd; ii != ei; ++ii)
	{
		GraphNode graphNode = graph->getEdgeDst(ii,Galois::MethodFlag::NONE);

		int nr_of_incoming_edges = atomic_dec(&graphNode->data.nr_of_incoming_edges);

		if(!nr_of_incoming_edges)
			ctx.push(graphNode);
	}

}

std::vector<Vertex*> *collectLeafs(Vertex *p)
{
	std::vector<Vertex *> *left = NULL;
	std::vector<Vertex *> *right = NULL;
	std::vector<Vertex*> *result = NULL;

	if (p == NULL) {
		return NULL;
	}

	result = new std::vector<Vertex*>();

	if (p!=NULL && p->right==NULL && p->left==NULL) {
		result->push_back(p);
		return result;
	}

	if (p!=NULL && p->left!=NULL) {
		left = collectLeafs(p->left);
	}

	if (p!=NULL && p->right!=NULL) {
		right = collectLeafs(p->right);
	}

	if (left != NULL) {
		for (std::vector<Vertex*>::iterator it = left->begin(); it!=left->end(); ++it) {
			if (*it != NULL) {
				result->push_back(*it);
			}
		}
		delete left;
	}

	if (right != NULL) {
		for (std::vector<Vertex*>::iterator it = right->begin(); it!=right->end(); ++it) {
			if (*it != NULL) {
				result->push_back(*it);
			}
		}
		delete right;
	}

	return result;
}

std::vector<double> *ProductionProcess::operator()(TaskDescription &taskDescription)
{
	// implement everything is needed to input data to solver,
	// preprocessing,
	srand(0xfafa);
	ITripleArgFunction *function = new TestFunction3D();
	GraphGenerator* generator = new GraphGenerator();
	AbstractProduction *production = new AbstractProduction(19, 75, 117, 83);
	Vertex *S;

	D3::MatrixGenerator *matrixGenerator = new D3::MatrixGenerator();

	Galois::StatTimer timerMatrix("matrix generation");
	timerMatrix.start();
	std::list<D3::Tier*> *tiers = matrixGenerator->CreateMatrixAndRhs(taskDescription.nrOfTiers,
			taskDescription.x, taskDescription.y, taskDescription.z, taskDescription.size, function);
	timerMatrix.stop();

	Processing *processing = new Processing();

	Galois::StatTimer timerPreprocess("preprocessing");

	timerPreprocess.start();
	std::vector<EquationSystem *> *inputMatrices = processing->preprocess((std::list<EquationSystem*> *)tiers,
		production);
	timerPreprocess.stop();


	Galois::StatTimer timerGraphGeneration("Graph generation");
	timerGraphGeneration.start();
	S = generator->generateGraph(taskDescription.nrOfTiers, production, inputMatrices);
	timerGraphGeneration.stop();

	Galois::StatTimer timerProductions("productions");
	timerProductions.start();
	graph = generator->getGraph();

	std::vector<GraphNode> initial_nodes_vector;
	for(LCM_iterator it = graph->begin(); it != graph->end(); ++it) {
		GraphNode graphNode = *(it);
		if(graphNode->data.nr_of_incoming_edges == 0)
			initial_nodes_vector.push_back(graphNode);
	}

	std::vector<GraphNode>::iterator iii = initial_nodes_vector.begin();
	typedef Galois::WorkList::dChunkedLIFO<1> WL;
	Galois::for_each<WL>(initial_nodes_vector.begin(), initial_nodes_vector.end(), *this);
	timerProductions.stop();


	std::vector<Vertex*> *leafs = collectLeafs(S);
	std::vector<double> *result = processing->postprocess(leafs, inputMatrices, production);

	if (taskDescription.performTests) {

		std::map<int, double> *mapa = new std::map<int, double>();
		int i = 0;

		for (std::vector<double>::iterator it=result->begin(); it!=result->end(); ++it, ++i) {
			(*mapa)[i] = *it;
		}

		matrixGenerator->checkSolution(mapa, function);

		mapa->clear();
		delete mapa;
	}

	delete leafs;
	delete processing;
	delete S;
	delete tiers;


	return result;
}

inline int ProductionProcess::atomic_dec(int *value) {
	// XXX: more portable solution?
	return __sync_add_and_fetch(value, -1);
}
