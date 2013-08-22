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
//	std::cout << "Node: " << src->data.x;
//	std::cout << " Production: ";
	switch (node.productionToExecute) {
	case A1:
		node.productions->A1(node.v, node.input);
		//std::cout << "A1" << std::endl;
		break;
	case A:
		node.productions->A(node.v, node.input);
		//std::cout << "A" << std::endl;
		break;
	case AN:
		node.productions->AN(node.v, node.input);
		//std::cout << "AN" << std::endl;
		break;
	case A2:
		node.productions->A2(node.v);
		//std::cout << "A2" << std::endl;
		break;
	case E:
		node.productions->E(node.v);
		//std::cout << "E" << std::endl;
		break;
	case EROOT:
		node.productions->ERoot(node.v);
		//std::cout << "EROOT" << std::endl;
		break;
	case BS:
		node.productions->BS(node.v);
		//std::cout << "BS" << std::endl;
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
			//(*this)(graphNode, ctx);
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

std::vector<double> *ProductionProcess::operator()(int nrOfTiers)
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
	std::list<D3::Tier*> *tiers = matrixGenerator->CreateMatrixAndRhs(nrOfTiers, -1e10, -1e10, -1e10, 1e+11, function);
	timerMatrix.stop();

	Processing *processing = new Processing();

	Galois::StatTimer timer_preproc("preprocessing");

	timer_preproc.start();
	std::vector<EquationSystem *> *inputMatrices = processing->preprocess((std::list<EquationSystem*> *)tiers,
		production);
	timer_preproc.stop();


	Galois::StatTimer timer_S("Graph generation");
	timer_S.start();
	S = generator->generateGraph(nrOfTiers, production, inputMatrices);
	timer_S.stop();

	Galois::StatTimer T("productions");
	T.start();
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
	typedef Galois::WorkList::dChunkedLIFO<1> WL;
	Galois::for_each<WL>(initial_nodes_vector.begin(), initial_nodes_vector.end(), *this);
	T.stop();

	printf("%f[s]\n", T.get()/1000.0);

	std::vector<Vertex*> *leafs = collectLeafs(S);
	std::vector<double> *result = processing->postprocess(leafs, inputMatrices, production);
	std::map<int, double> *mapa = new std::map<int, double>();

	int i = 0;

	for (std::vector<double>::iterator it=result->begin(); it!=result->end(); ++it, ++i) {
		(*mapa)[i] = *it;
	}
	matrixGenerator->checkSolution(mapa, function);

	mapa->clear();

	/*
	for (std::vector<double>::iterator it=result->begin(); it!=result->end(); ++it) {
		printf("%.16g\n", *it);
	}
	 */
	delete leafs;
	delete processing;
	delete S;
	delete tiers;
	//delete matrixGenerator;

	delete mapa;

	return result;
}

inline int ProductionProcess::atomic_dec(int *value) {
	// XXX: more portable solution?
	return __sync_add_and_fetch(value, -1);
}
