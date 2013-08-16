#include "GaloisWorker.h"
#include "Point3D_tmp/MatrixGenerator.hxx"
#include <vector>
#include "Point3D_tmp/TripleArgFunction.hxx"
#include "Postprocessor.h"
/*
class TestFunction : public IDoubleArgFunction {
	double ComputeValue(double x, double y) {
		return x*x+y*y;
	}
}; */

using namespace tmp;

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
	std::vector<Vertex*> *result;

	if (p!=NULL && p->left!=NULL) {
		left = collectLeafs(p->left);
	}
	if (p!=NULL && p->right!=NULL) {
		right = collectLeafs(p->right);
	}
	if (p!=NULL && p->right==NULL && p->left==NULL) {
		result = new std::vector<Vertex*>(1);
		result->push_back(p);
		return result;
	}
	if (p!= NULL) {
		result = new std::vector<Vertex*>();
	} else {
		result = NULL;
	}
	if (left != NULL) {
		for (std::vector<Vertex*>::iterator it = left->begin(); it!=left->end(); ++it) {
			result->push_back(*it);

		}
		delete left;
	}
	if (right != NULL) {
		for (std::vector<Vertex*>::iterator it = right->begin(); it!=right->end(); ++it) {
			result->push_back(*it);
		}
		delete right;
	}

	return result;

}

std::vector<double> *ProductionProcess::operator()(int nrOfTiers)
{
	// implement everything is needed to input data to solver,
	// preprocessing,
	ITripleArgFunction *function = new TestFunction3D();
	GraphGenerator* generator = new GraphGenerator();
	AbstractProduction *production = new AbstractProduction(19, 75, 117, 83);
	Vertex *S;
	MatrixGenerator *matrixGenerator = new MatrixGenerator();
	std::list<tmp::Tier*> *tiers = matrixGenerator->CreateMatrixAndRhs(nrOfTiers, 0, 0, 0, 1, function);
	Mes3DPreprocessor *preprocessor = new Mes3DPreprocessor();
	std::vector<EquationSystem *> *inputMatrices = preprocessor->preprocess(tiers);
	S = generator->generateGraph(nrOfTiers, production, inputMatrices);

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

	std::vector<Vertex*> *leafs = collectLeafs(S);
	Postprocessor3D *mes3dProcessor = new Postprocessor3D();
	std::vector<double> *result = mes3dProcessor->postprocess(leafs, inputMatrices, production);

	delete leafs;
	delete mes3dProcessor;
	delete S;

	return result;
}

inline int ProductionProcess::atomic_dec(int *value) {
	// XXX: more portable solution?
	return __sync_add_and_fetch(value, -1);
}
