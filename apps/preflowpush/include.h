#include "Galois/Launcher.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Node.h"
#include "Edge.h"


typedef Galois::Graph::FirstGraph<Node,Edge,false>            Graph;
typedef Galois::Graph::FirstGraph<Node,Edge,false>::GraphNode GNode;


#include "Local.h"
int numNodes;
int numEdges;
Graph* config;
GNode sink;
GNode source;

#include "Builder.h"
#include "Support/ThreadSafe/simple_lock.h"
#include "Support/ThreadSafe/TSStack.h"

void  initializePreflow(vector<int>& gapYet);
void process(GNode item, threadsafe::ts_stack<GNode>& lwl);
bool discharge(threadsafe::ts_stack<GNode>& lwl , vector<int>& gapYet, GNode& src) ;
void relabel(GNode& src, Local& l);
void globalRelabelSerial(vector<int>& gapYet, threadsafe::ts_stack<GNode>& lw);
threadsafe::ts_stack<GNode> wl;
int threads = 1;
vector<int> gapYet;

const int ALPHA=6;
const int BETA=12;
const int globalRelabelInterval = numNodes * ALPHA + numEdges;
int relabelYet;
