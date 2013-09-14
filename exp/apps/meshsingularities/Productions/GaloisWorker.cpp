#include <vector>

#include "GaloisWorker.h"
#include "Processing.h"
#include "PointProduction.hxx"
#include "EdgeProduction.h"
#include "Point2D/MatrixGenerator.hxx"
#include "Point3D/MatrixGenerator.hxx"
#include "Point2DQuad/MatrixGenerator.hxx"
#include "Edge2D/MatrixGenerator.hxx"

#include "FakeMatrixGenerator.h"

#include "Node.h"

//Galois::Runtime::LL::SimpleLock<true> foo;

template<typename Context>
void ProductionProcess::operator()(Graph::GraphNode src, Context& ctx)
{
	//unsigned tid = Galois::Runtime::LL::getTID();

	Node &node = src->data;
	// node-related work is here:
	node.execute();

	for(LCM_edge_iterator ii = src->edgeBegin, ei = src->edgeEnd; ii != ei; ++ii)
	{
		GraphNode graphNode = graph->getEdgeDst(ii,Galois::MethodFlag::NONE);
		int nr_of_incoming_edges = atomic_dec(&(graphNode->data.incomingEdges));

		if(!nr_of_incoming_edges)
			ctx.push(graphNode);
	}
}

int ProductionProcess::leftRange(int tasks, int cpus, int i)
{
	if (i == 0) {
		return 0;
	}
	return rightRange(tasks, cpus, i-1);
}

int ProductionProcess::rightRange(int tasks, int cpus, int i)
{
	return ((i+1)*tasks)/cpus + ((i < (tasks % cpus)) ? 1 : 0) - 1;
}

std::vector<double> *ProductionProcess::operator()(TaskDescription &taskDescription)
{

	// implement everything is needed to input data to solver,
	// preprocessing,
	//srand(0xfafa);

	AbstractProduction *production;
	Vertex *S;
	Galois::StatTimer TMain;
	TMain.start();

	GenericMatrixGenerator* matrixGenerator;

	if(taskDescription.dimensions == 3) {
		switch (taskDescription.singularity) {
		case POINT:
			matrixGenerator = new D3::MatrixGenerator();
			break;
		case CENTRAL_POINT:
			matrixGenerator = new PointCentral3DMatrixGenerator();
			break;
		case EDGE:
			matrixGenerator = new Edge3DMatrixGenerator();
			break;
		case FACE:
			matrixGenerator = new Face3DMatrixGenerator();
			break;
		case ANISOTROPIC:
			matrixGenerator = new Anisotropic3DMatrixGenerator();
			break;

		}
	}
	else if (taskDescription.dimensions == 2 && !taskDescription.quad) {
		switch (taskDescription.singularity) {
		case POINT:
			matrixGenerator = new D2::MatrixGenerator();
			break;
		case CENTRAL_POINT:
			matrixGenerator = new PointCentral2DMatrixGenerator();
			break;
		case EDGE:
			matrixGenerator = new D2Edge::MatrixGenerator();
			break;
		default:
			printf("Error: unknown type of singularity in 2D!\n");
			exit(1);
			break;
		}
		//matrixGenerator = new D2::MatrixGenerator();
		//production = new AbstractProduction(5, 17, 21, 21);
	}

	/*production = new AbstractProduction(matrixGenerator->getiSize(taskDescription.polynomialDegree),
			matrixGenerator->getLeafSize(taskDescription.polynomialDegree),
			matrixGenerator->getA1Size(taskDescription.polynomialDegree),
			matrixGenerator->getANSize(taskDescription.polynomialDegree));*/
	bool edge = taskDescription.dimensions == 2 && !taskDescription.quad && taskDescription.singularity == EDGE;
	if(edge)
		printf("EDGE SOLVER!\n");
	std::vector<int>* vec = matrixGenerator->GetProductionParameters(taskDescription.polynomialDegree);
	if(edge)
		production = new EdgeProduction(vec);
	else
		production = new PointProduction(vec);


	/*printf("Problem size: %d\n", matrixGenerator->getA1Size(taskDescription.polynomialDegree) +
			matrixGenerator->getANSize(taskDescription.polynomialDegree)
			+ matrixGenerator->getLeafSize(taskDescription.polynomialDegree)*taskDescription.nrOfTiers
			- taskDescription.nrOfTiers*matrixGenerator->getiSize(taskDescription.polynomialDegree));*/

	Galois::StatTimer timerMatrix("MATRIX GENERATION");
	timerMatrix.start();
	std::list<EquationSystem*> *tiers = matrixGenerator->CreateMatrixAndRhs(taskDescription);
	timerMatrix.stop();

	Processing *processing = new Processing();
	std::vector<EquationSystem *> *inputMatrices;
	if(edge)
	{
		inputMatrices = new std::vector<EquationSystem*>();
		for(std::list<EquationSystem*>::iterator it = tiers->begin(); it != tiers->end(); ++it)
		{
			inputMatrices->push_back(*it);
		}
	}
	else
	{
		Galois::StatTimer timerPreprocess("PREPROCESSING");

		timerPreprocess.start();
		inputMatrices = processing->preprocess((std::list<EquationSystem*> *)tiers,
				(PointProduction*)production);
		timerPreprocess.stop();
	}

	S = production->getRootVertex();

	Galois::StatTimer timerProductions("PRODUCTIONS");
	timerProductions.start();
	graph = production->getGraph();

	std::vector<GraphNode> initial_nodes_vector;
	for(LCM_iterator it = graph->begin(); it != graph->end(); ++it) {
		GraphNode graphNode = *(it);
		if(graphNode->data.incomingEdges == 0) {
			initial_nodes_vector.push_back(graphNode);
		}
	}

	std::vector<GraphNode>::iterator iii = initial_nodes_vector.begin();

	/*const int maxPackages = Galois::Runtime::LL::getMaxPackages();
	const int coresPerPackages = Galois::Runtime::LL::getMaxCores()/maxPackages;

	const int activePackages = std::ceil(Galois::Runtime::activeThreads*1.0/coresPerPackages);

	printf("Active threads: %d\n",Galois::Runtime::activeThreads);
	printf("Max cores: %d\n", Galois::Runtime::LL::getMaxCores());
	printf("Active packages: %d\n", activePackages);
	const int tasks = initial_nodes_vector.size();

	for (int pack=0; pack < activePackages; ++pack) {
		std::vector<GraphNode> packNodes;
		printf("%d -> %d\n", leftRange(tasks, activePackages, pack), rightRange(tasks, activePackages, pack));
		for (int i=leftRange(tasks, activePackages, pack); i<rightRange(tasks, activePackages, pack); ++i) {
			packNodes.push_back(*iii);
			++iii;
			printf("adding %d to %d\n", i, pack);

		}
		pps.getRemoteByPkg(pack)->
				push(packNodes.begin(), packNodes.end());

	} */


	//printf ("PerPackageStorage size: %d\n", pps.size());
	Galois::for_each<WL>(initial_nodes_vector.begin(), initial_nodes_vector.end(), *this);
	//Galois::do_all(initial_nodes_vector.begin(), initial_nodes_vector.end(), *this);
	//for (;iii != initial_nodes_vector.end(); ++iii) {
	//	(*this)(*iii);
	//}

	//Galois::on_each([&] (unsigned tid, unsigned totalThreads) {
	//	printf("Hello from: [%d, %d]\n", tid, totalThreads);
	//	Galois::Runtime::getSystemTermination().initializeThread();
	//	if (Galois::Runtime::LL::isPackageLeaderForSelf(tid)) {
	//		do {
	//			printf("TID: %d is leader\n", tid);
	//			GraphNode graphNode = pps.getLocal(tid)->pop().get();
	//
	//			Node &node = graphNode->data;
	//			node.execute();
	//		} while(!Galois::Runtime::getSystemTermination().globalTermination());

	//	}
	//	Galois::Runtime::getSystemTermination().localTermination(true);
	//});

	timerProductions.stop();
	std::vector<double> *result;
	if(!edge)
	{
		std::vector<Vertex*> *leafs = processing->collectLeafs(S);
		result = processing->postprocess(leafs, inputMatrices, (PointProduction*)production);
		delete leafs;
	}
	else
	{
		result = new std::vector<double>();
	}

	if (taskDescription.performTests && !edge) {

		std::map<int, double> *mapa = new std::map<int, double>();
		int i = 0;

		for (std::vector<double>::iterator it=result->begin(); it!=result->end(); ++it, ++i) {
			(*mapa)[i] = *it;
		}

		matrixGenerator->checkSolution(mapa, taskDescription.function);

		mapa->clear();
		delete mapa;
	}


	delete vec;
	delete processing;
	delete S;
	delete tiers;

	TMain.stop();

	return result;
}

inline int ProductionProcess::atomic_dec(int *value) {
	// XXX: more portable solution?
	return __sync_sub_and_fetch(value, 1);
}
