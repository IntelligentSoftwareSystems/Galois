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

#include "galois/MethodFlags.h"

#include "Node.h"

//galois::runtime::LL::SimpleLock<true> foo;

template<typename Context>
void ProductionProcess::operator()(Graph::GraphNode src, Context& ctx)
{

	Node &node = src->getData();
	// node-related work is here:
	node.execute();

	for(LCM_edge_iterator ii = graph->edge_begin(src, galois::MethodFlag::UNPROTECTED), ei = graph->edge_end(src, galois::MethodFlag::UNPROTECTED); ii != ei; ++ii)
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

void printVertices(Vertex *v, std::string space)
{
    printf("%ssize: %d x %d\n", space.c_str(), v->system->n, v->system->n);
    if (v->left != NULL) {
        printVertices(v->left, space+" ");
    }
    if (v->right != NULL) {
        printVertices(v->right, space+" ");
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
	AbstractProduction *production;
	Vertex *S;
	galois::StatTimer TMain;
    //TMain.start();
	
#ifdef WITH_PAPI
	long long fpops = 0;
	bool papi_supported = true;
	int events[1] = {PAPI_FP_OPS};
    int papi_err;
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI is unsupported.\n");
        papi_supported = false; 
    }

    if (PAPI_num_counters() < 2) {
        fprintf(stderr, "PAPI is unsupported.\n");
        papi_supported = false;
    }
#endif

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



	galois::StatTimer timerMatrix("MATRIX GENERATION");
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

	galois::StatTimer timerSolution("SOLUTION");

	struct timeval start_time; 
	struct timeval end_time; 

    if(edge)
    	production = new EdgeProduction(vec, inputMatrices);
    else
        production = new PointProduction(vec, inputMatrices);

	S = production->getRootVertex();
    //printVertices(S, "");
	printf("Allocated: %lu bytes \n", this->getAllocatedSize(S));
    printf("Root size: %d\n", S->system->n);
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

#ifdef WITH_PAPI
    if (papi_supported) {
        if ((papi_err = PAPI_start_counters(events, 1)) != PAPI_OK) {
            fprintf(stderr, "Could not start counters: %s\n", PAPI_strerror(papi_err));
        }
    }
#endif
    if (taskDescription.scheduler == OLD) {
        galois::for_each(initial_nodes_vector.begin(), initial_nodes_vector.end(), *this, galois::wl<WL>());
    } else if (taskDescription.scheduler == CILK) {
        // TODO: implement CILK

    } else {
        // TODO: implement Galois-DAG
    }
#ifdef WITH_PAPI	
	if (papi_supported) {
    	if ((papi_err = PAPI_read_counters(&fpops, 1)) != PAPI_OK) {
            fprintf(stderr, "Could not get values: %s\n", PAPI_strerror(papi_err));
        }
        printf("FLOPS: %ld\n", fpops);
    }
#endif

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

    //TMain.stop();

	return result;
}

unsigned long ProductionProcess::getAllocatedSize(Vertex *root)
{
    unsigned long total = 0;
    if (root != NULL) {
        total = (root->system->n+1)*root->system->n*sizeof(double);
        total += getAllocatedSize(root->left)+getAllocatedSize(root->right);
    }
    return total;
}

inline int ProductionProcess::atomic_dec(int *value) {
	// XXX: more portable solution?
	return __atomic_sub_fetch(value, 1, __ATOMIC_ACQ_REL);
}
