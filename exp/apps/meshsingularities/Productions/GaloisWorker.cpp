#include <vector>

#include "GaloisWorker.h"
#include "PointProduction.hxx"
#include "EdgeProduction.h"
#include "Point2D/MatrixGenerator.hxx"
#include "Point3D/MatrixGenerator.hxx"
#include "Point2DQuad/MatrixGenerator.hxx"
#include "Edge2D/MatrixGenerator.hxx"
#include <sys/time.h>
#include "FakeMatrixGenerator.h"

#include "Galois/MethodFlags.h"

#include "Node.h"
//Galois::Runtime::LL::SimpleLock<true> foo;

template<typename Context>
void ProductionProcess::operator()(Graph::GraphNode src, Context& ctx)
{

	Node &node = src->getData();
	// node-related work is here:
	node.execute();

	for(LCM_edge_iterator ii = graph->edge_begin(src, Galois::MethodFlag::NONE), ei = graph->edge_end(src, Galois::MethodFlag::NONE); ii != ei; ++ii)
	{
		GraphNode graphNode = graph->getEdgeDst(ii);
		int nr_of_incoming_edges = atomic_dec(&(graphNode->getData().incomingEdges));

		if(!nr_of_incoming_edges)
			ctx.push(graphNode);
	}
}

double test_function(int dim, ...)
{
	double *data = new double[dim];
	double result = 0;
	va_list args;

	va_start (args, dim);
	for (int i=0; i<dim; ++i) {
		data[i] = va_arg (args, double);
	}
	va_end(args);

	if (dim == 2)
	{

		result = 1;
		//result = data[0]*data[1]+data[0]+data[0]*data[1]*data[0]*data[1] + 11;
	}
	else
	{
		result = -1;
	}

	delete [] data;
	return result;
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
	else if (taskDescription.dimensions == 2) {
		switch (taskDescription.singularity) {
		case POINT:
			matrixGenerator = new D2::MatrixGenerator();
			break;
		case CENTRAL_POINT:
			matrixGenerator = new PointCentral2DMatrixGenerator();
			break;
		case EDGE:
			if (taskDescription.quad)
				matrixGenerator = new D2Edge::MatrixGenerator();
			else
				matrixGenerator = new Edge2DMatrixGenerator();
			break;
		default:
			printf("Error: unknown type of singularity in 2D!\n");
			exit(1);
			break;
		}
		//matrixGenerator = new D2::MatrixGenerator();
		//production = new AbstractProduction(5, 17, 21, 21);
	}

	bool edge = taskDescription.dimensions == 2 && taskDescription.quad && taskDescription.singularity == EDGE;



	Galois::StatTimer timerMatrix("MATRIX GENERATION");
	timerMatrix.start();
	std::vector<EquationSystem*> *tiers = matrixGenerator->CreateMatrixAndRhs(taskDescription);

	//parameters are ready after matrix creation
	std::vector<int>* vec = matrixGenerator->GetProductionParameters(taskDescription.polynomialDegree);
	if (!edge)
		printf("Problem size: %d\n", (*vec)[3] +
                        (*vec)[2]
                       	+ (*vec)[1]*(taskDescription.nrOfTiers-2)
                        - (taskDescription.nrOfTiers-1)* (*vec)[0]);
	else
		printf("Problem size: %d\n", (*vec)[0]);

	timerMatrix.stop();

	std::vector<EquationSystem *> *inputMatrices;
	inputMatrices = tiers;

	Galois::StatTimer timerSolution("SOLUTION");

	struct timeval start_time; 
	struct timeval end_time; 



    if(edge)
    	production = new EdgeProduction(vec, inputMatrices);
    else
        production = new PointProduction(vec, inputMatrices);

	S = production->getRootVertex();
	timerSolution.start();
	int xx = gettimeofday(&start_time, NULL);
	graph = production->getGraph();

	std::vector<GraphNode> initial_nodes_vector;
	for(LCM_iterator it = graph->begin(); it != graph->end(); ++it) {
		GraphNode graphNode = *(it);
		if(graphNode->getData().incomingEdges == 0) {
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


	Galois::for_each(initial_nodes_vector.begin(), initial_nodes_vector.end(), *this, Galois::wl<WL>());
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

	timerSolution.stop();
	printf("SOLUTION READY\n"); 
	xx += gettimeofday(&end_time, NULL);
	if(xx == 0)
	{
       		printf("time %f [s]\n", ((end_time.tv_sec - start_time.tv_sec)*1000000 +(end_time.tv_usec - start_time.tv_usec))/1000000.0);
	}

	std::vector<double> *result;
	if(!edge)
	{
		result = ((PointProduction*)production)->getResult();
	}
	else
	{
		//edge tiers check their solution based on local numeration
		result = ((EdgeProduction*)production)->getResult();
	}

	if (taskDescription.performTests) {
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
	delete S;
	delete tiers;

	TMain.stop();

	return result;
}

unsigned long ProductionProcess::getAllocatedSize(Vertex *root)
{
    unsigned long total = 0;
    if (root != NULL) {
        unsigned long total = (root->system->n+1)*root->system->n*sizeof(double);
        total += getAllocatedSize(root->left)+getAllocatedSize(root->right);
    }
    return total;
}

inline int ProductionProcess::atomic_dec(int *value) {
	// XXX: more portable solution?
	return __atomic_sub_fetch(value, 1, __ATOMIC_ACQ_REL);
}
