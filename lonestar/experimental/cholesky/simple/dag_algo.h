//
// dag_algo.h - Run an operator on a graph using a DAG scheduler
//

#include <stdio.h>
#include <sys/time.h>

/** Our work queue */
pthread_barrier_t thread_init_barrier;

template <typename T, typename Graph>
struct thread_args {
  T* WL;
  Graph* graph;
};

template <typename T, typename Graph>
void* thread_main(void* arg) {
  struct thread_args<T, Graph>* args = (struct thread_args<T, Graph>*)arg;
  T* WL                              = args->WL;
  Graph* graph                       = args->graph;
  node_t node;

  // All threads start at same time
  pthread_barrier_wait(&thread_init_barrier);

  while ((node = WL->pop()) != INVALID) {
    do_node(graph, node);
    /* Update indegree of outgoing edges; add ready neighbors to worklist. */
    int pushed = 0;
    for (edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
         ii != ei; ii++) {
      node_t neighbor = graph->edge_dest(node, ii);
      if (neighbor == node)
        continue;
      unsigned int nindegree =
          __sync_sub_and_fetch(&graph->_node_data(neighbor).indegree, 1);
      assert(nindegree <= graph->nodecount);
      if (nindegree == 0) {
        WL->push(neighbor); /* Add them to the worklist */
        pushed++;
      }
    }
    // printf("Pushed %d nodes\n", pushed);
  }
  printf("Thread %p done\n", (void*)pthread_self());
  // All threads end at same time
  pthread_barrier_wait(&thread_init_barrier);
  return NULL;
}

template <typename T, typename Graph>
void run_algo(T* WL, Graph* graph, unsigned nthreads) {
  /* Create initial worklist */
  {
    int npushed = 0;
    for (node_t node = 0; node < graph->nodecount; node++) {
      if (graph->_node_data(node).indegree > 0)
        continue;
      WL[npushed % nthreads].push(node);
      npushed++;
    }
    assert(npushed > 0 || graph->nodecount <= 0);
    printf("%d nodes added to worklist\n", npushed);
  }

  /* Launch threads */
  pthread_t threads[nthreads];
  struct thread_args<T, Graph> thread_args[nthreads];
  for (unsigned i = 0; i < nthreads; i++) {
    thread_args[i].WL    = &WL[i];
    thread_args[i].graph = graph;
    if (pthread_create(&threads[i], NULL, thread_main<T, Graph>,
                       &thread_args[i]))
      abort();
  }
  /* Reap threads */
  for (unsigned i = 0; i < nthreads; i++)
    if (pthread_join(threads[i], NULL))
      abort();
}

template <typename Graph>
struct NodeCmp {
  Graph* graph;
  int multiplier;

public:
  NodeCmp(Graph* graph, const int multiplier = 1)
      : graph(graph), multiplier(multiplier) {}
  bool operator()(const node_t x, const node_t y) const {
    // printf("cmp: %u<%u => %u<%u -- *%d\n",x, y,
    //       G->nodedata[x],G->nodedata[y], multiplier);
    unsigned int xdegree = graph->_node_data(x).outdegree,
                 ydegree = graph->_node_data(y).outdegree;
    return multiplier > 0 ? (xdegree < ydegree) : (xdegree > ydegree);
  }
};

/* Time measurement (from HW1, HW3) */
#define TIMEDELTA(a, b, small, multiplier)                                     \
  (((b).tv_sec - (a).tv_sec) * multiplier + ((b).small - (a).small))
#define SEC_IN_MICROSEC 1000000
#define TIMEDELTA_MICRO(a, b) TIMEDELTA(a, b, tv_usec, SEC_IN_MICROSEC)

template <typename Graph>
void run_dag(Graph* graph, unsigned nthreads) {
  printf("Using %d threads\n", nthreads);

  if (pthread_barrier_init(&thread_init_barrier, NULL, nthreads))
    abort();

  // Create worklists
  typedef NodeCmp<Graph> Compare;
  typedef LocalWorklist<node_t, INVALID, Compare> LW;
  typename LW::threadlist tobjs;
  tobjs.reserve(nthreads);
  for (unsigned i = 0; i < nthreads; i++) {
    LW tobj(&tobjs, i, Compare(graph, 1));
    tobjs.push_back(tobj);
  }
  tobjs[0].set_token(0);

  // Launch threads
  struct timeval start, end;
  gettimeofday(&start, NULL);
  run_algo(&tobjs[0], graph, nthreads);
  gettimeofday(&end, NULL);
  printf("%d threads done in %ld ms\n", nthreads,
         TIMEDELTA_MICRO(start, end) / 1000);
}
