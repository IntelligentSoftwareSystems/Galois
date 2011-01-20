#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Node.h"
#include "Edge.h"


typedef Galois::Graph::FirstGraph<Node,Edge,true>            Graph;
typedef Galois::Graph::FirstGraph<Node,Edge,true>::GraphNode GNode;

class Index
{
	public:
		int index_func(GNode n )
		{
			return n.getData(Galois::Graph::NONE,0).height;
		}
};

#include "Local.h"
int numNodes;
int numEdges;
Graph* config;
GNode sink;
GNode source;

#include "Builder2.h"
//#include "PQ.h"

void  initializePreflow();
void process(GNode& item , Galois::Context<GNode>& lwl);

template<typename Context>
void relabel(GNode& src, Local& l, Context* cnx);

template<typename Context>
bool discharge( GNode& src,Context* cnx);

template<typename Context>
void bfs(GNode& src, Context* cnx);

template<typename Context>
void globalRelabelSerial(Context* cnx);

void checkMaxFlow();
void checkFlowsForCut();
void checkHeights();
void checkAugmentingPathExistence();

int threads = 1 ;
vector<int> gapYet;
double expected;

//vector_wl<GNode> wl;
//threadsafe::ts_pqueue< GNode, compare > wl;
//ordered_wl<GNode, Index > wl;

//threadsafe::ts_queue<GNode> que;
GaloisRuntime::WorkList::FIFO<GNode> que;
std::vector<GNode> wl;
//threadsafe::ts_queue<GNode> wl;

volatile int counter=0;
volatile bool lock=false;
volatile long int token=0;
const int ALPHA=6;
const int BETA=12;
int globalRelabelInterval;
volatile int relabelYet;
struct timeval tvBegin, tvEnd, tvDiff;
long int global_sec=0,global_usec=0;

