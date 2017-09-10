/* g++ -O2 -g -L /workspace/ATLAS/Linux_nordstrom/lib -I /workspace/ATLAS/include dagcholesky.cpp -lcblas -latlas -lpthread */

#include <unistd.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <string.h>

#include "simplepapi.h"

extern "C" {
#include <cblas.h>              /* BLAS */
}

typedef struct nodedata {
  int seen;
#ifdef EBN
  std::vector<std::tuple<unsigned, unsigned, unsigned> > ebn; // ii, ij, bridge
#endif
#ifdef ROWNAV
  unsigned order;
#endif
} nodedata_t;
typedef struct edgedata {
  double data;
  //char lock;
  pthread_mutex_t lock;
#ifdef ROWNAV
  unsigned rownext_node, rownext_edge;
#endif
} edgedata_t;

#include "dag.h"

void init_node(node_t id, nodedata_t *data) {
  data->seen = 0;
}

void init_edge(node_t from, node_t to, edgedata_t *data, char *extra) {
  /* Load edge weight (if data is non-NULL) */
  data->data = extra ? atof(extra) : 0; // 0 should never happen
  //data.lock = 0;
  if ( pthread_mutex_init(&data->lock, NULL) ) abort();
}

//static unsigned counter = 0;

template <typename Graph>
void do_node(Graph *graph, node_t node) {
  {
    nodedata_t *nd = graph->node_data(node);
    // Update seen flag on node
    assert(nd->seen == 0);
    // Shouldn't need an atomic for this -- especially since it doesn't
    // depend on the previous value. Protected by DAG schedule (no more
    // incoming edges).
    nd->seen = 1;
  }

  //std::cout << "STARTING " << node->id << "\n";

  // Find self-edge for this node, update it
  edge_t selfedge = graph->find_edge(node, node);
  assert(selfedge != INVALID);
  double *factor_p = &graph->edge_data(node, selfedge)->data, factor = *factor_p;
  //printf("STARTING %4d %10.5f\n", node, factor);
  assert(factor > 0);
  // Shouldn't need an atomic for this. DAG schedule should
  // ensure not used as bridge while processing source node.
  *factor_p = factor = sqrt(factor);
  assert(factor != 0 && !isnan(factor));

  //std::cout << "STARTING " << noded.id << " " << factor << "\n";
  //printf("STARTING %4d %10.5f\n", noded.id, factor);

  // Dense support: build a vector of edgeweights
#if 0
  unsigned nedges = graph->_node_data(node).outdegree, k = 0; // FIXME
  double edgevec[nedges], edgemat[nedges*nedges]; // Stack-stored, not heap
  int useedgevec = 0;//nedges > 500;
  //printf("EDGEMAT: %u\n", nedges*nedges);
#endif

  // Update all edges (except self-edge)
  for ( edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
        ii != ei; ii++ ) {
    node_t dest = graph->edge_dest(node, ii);
    nodedata_t *destdata = graph->node_data(dest);
    if ( destdata->seen == 0 ) {
      // Shouldn't need an atomic here, either. DAG schedule should
      // ensure not used as bridge while processing source node.
      double *edgedata_p = &graph->edge_data(node, ii)->data;
      *edgedata_p /= factor;
#if 0
      if ( useedgevec ) {
        edgevec[k] = *edgedata_p;
        k++;
      }
#endif

      //printf("N-EDGE %4d %4d %10.5f\n", node->id, dst->id, edge->data);
      //std::cout << noded.id << " " << dstd.id << " " << ed << "\n";
    }
  }

#if 0
  if ( useedgevec ) {
    assert(k == nedges-1);
    // Compute the products for the bridge edgeweights
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nedges, nedges, 1,
                1.0, edgevec, 1, edgevec, nedges, 0.0, edgemat, nedges);
    k = 0;
  }
#endif

  // Update all edges between neighbors (we're operating on the filled graph,
  // so we they form a (directed) clique)
#ifdef ROWNAV
  {
    // Scatter the column into a dense vector for use during row traversal
    double column[graph->nodecount];
    memset(column, 0, graph->nodecount*sizeof(double));
    for ( edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
          ii != ei; ii++ ) {
      node_t src = graph->edge_dest(node, ii);
      if ( graph->node_data(src)->seen == 0 )
        column[src] = graph->edge_data(node, ii)->data;
    }
    // Traverse each row to update
    for ( edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
          ii != ei; ii++ ) {
      node_t src = node, dest = graph->edge_dest(node, ii);
      //counter++;
      //printf("STARTING ROW AT %u %u\n", src, dest);
      if ( column[dest] == 0 ) continue;
      edge_t bridge = ii;
      edgedata_t *bridgedata = graph->edge_data(src, bridge);
      while ( 1 ) {
        //printf("Checking rownext for %u %u\n", src, graph->edge_dest(src, bridge));
        src = bridgedata->rownext_node;
        bridge = bridgedata->rownext_edge;
        dest = graph->edge_dest(src, bridge);
        //printf("EDGE IS %d %d\n", src, dest);

        bridgedata = graph->edge_data(src, bridge);
        if ( column[src] != 0 && column[dest] != 0 ) {
          double diff = column[src]*column[dest];
          pthread_mutex_lock(&bridgedata->lock);
          assert(src != dest || diff < bridgedata->data);
          bridgedata->data -= diff;
          pthread_mutex_unlock(&bridgedata->lock);
          //printf("I-EDGE %4d %4d %4d %10.5f\n", node, src, dest, bridgedata->data);
        }
        //else printf("SKIPPING %4d %4d\n", src, dest);
        if ( dest == src ) {
          //printf("DONE WITH ROW %d\n", dest);
          break;
        }
      }
    }
  }
#else
#ifdef EBN
  for ( auto ik = graph->node_data(node)->ebn.begin(),
          ek = graph->node_data(node)->ebn.end(); ik != ek; ik++ ) {
    edge_t ii = std::get<0>(*ik), ij = std::get<1>(*ik),
      bridge = std::get<2>(*ik);
    assert(ii != INVALID && ij != INVALID && bridge != INVALID
#if 0
           && !useedgevec
#endif
           );
#else
  for ( edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
        ii != ei; ii++ ) {
#endif
    node_t src = graph->edge_dest(node, ii);
#if 0
    unsigned l = 0;
#endif
    if ( graph->node_data(src)->seen > 0 ) continue;

#ifndef EBN
    // Enumerate all other neighbors
    for ( edge_t ij = graph->edge_begin(node), ej = graph->edge_end(node);
          ij != ej; ij++ ) {
#endif
      node_t dest = graph->edge_dest(node, ij);
      if ( graph->node_data(dest)->seen > 0 ) continue;

#ifndef EBN
      // Find the edge that bridges these two neighbors
      edge_t bridge = graph->find_edge(src, dest);
      if ( bridge != INVALID ) {
#endif
        edgedata_t *bridgedata = graph->edge_data(src, bridge);
        //printf("B-EDGE %4d %4d %4d %10.5f\n", node, src, dest, bridgedata.data);
        // Update the weight of the bridge edge. This does need an atomic,
        // since multiple nodes can share a bridge between neighbors.
        //while (!__sync_bool_compare_and_swap(&(bridgedata.lock), 0, 1));
        double diff =
#if 0
          useedgevec ? edgemat[l+k*nedges] :
#endif
          (graph->edge_data(node, ii)->data*graph->edge_data(node, ij)->data);
#ifndef SERIAL
        pthread_mutex_lock(&bridgedata->lock);
#endif
        assert(src != dest || diff < bridgedata->data);
        //assert(bridgedata.lock == 1);
        bridgedata->data -= diff;
        //__sync_fetch_and_sub(&(bridgedata.data), diff);
        //__sync_synchronize(); // Niftier: volatile asm (""):::memory
        //bridgedata.lock = 0;
#ifndef SERIAL
        pthread_mutex_unlock(&bridgedata->lock);
#endif
        //if ( src == dest )
        //printf("I-EDGE %4d %4d %4d %10.5f\n", node, src, dest, bridgedata.data);
        //std::cout << srcd.id << " " << dstd.id << " " << edb << "\n";
#ifdef EBN
      }
#else
      }
#if 0
      if ( useedgevec ) l++;
#endif
    }
#if 0
    if ( useedgevec ) k++;
#endif
  }
#endif  // EBN
#endif  // ROWNAV
  //std::cout << "OPERATED ON " << node->id << "\n";
  //sleep(1); // Maybe use this to help debug parallelism
}

template <typename Graph>
void print_graph(Graph *graph) {
  /* Print all edges */
  FILE *fh = fopen("dagcholeskyedges.txt", "w");
  if ( !fh ) abort();
  for ( node_t node = 0; node < graph->nodecount; node++ ) {
    for ( edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
          ii != ei; ii++ ) {
      fprintf(fh, "%d %d %.*e\n", node, graph->edge_dest(node, ii),
              DBL_DIG+3, graph->edge_data(node, ii)->data);
    }
  }
  if ( fclose(fh) != 0 ) abort();
}

template <typename Graph>
void compute_ebn(Graph *graph, char *argv[]) {
#ifdef EBN
  for ( node_t node = 0; node < graph->nodecount; node++ ) {
    for ( edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
          ii != ei; ii++ ) {
      node_t src = graph->edge_dest(node, ii);
      for ( edge_t ij = graph->edge_begin(node), ej = graph->edge_end(node);
            ij != ej; ij++ ) {
        node_t dest = graph->edge_dest(node, ij);
        edge_t bridge = graph->find_edge(src, dest);
        if ( bridge != INVALID )
          graph->node_data(src)->ebn.push_back(std::make_tuple(ii, ij, bridge));
      }
    }
  }
#endif
#ifdef ROWNAV
  std::vector<unsigned> nodeorder;
  {
    FILE *fh = fopen(argv[2], "r");
    if ( !fh ) abort();
    unsigned x;
    while ( fscanf(fh, "%u", &x) == 1 ) {
      nodeorder.push_back(x);
    }
    fclose(fh);
  }
  if ( nodeorder.size() != graph->nodecount ) {
    abort();
  }

  // Initialize all rownext pointers to self-edge (at end of row)
  for ( node_t node = 0; node < graph->nodecount; node++ ) {
    for ( edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
          ii != ei; ii++ ) {
      node_t dest = graph->edge_dest(node, ii);
      graph->edge_data(node, ii)->rownext_node = dest;
      graph->edge_data(node, ii)->rownext_edge = graph->find_edge(dest, dest);
      //printf("INITIALIZING %u %u rownext to be %u %u (%u)\n", node, dest, dest, dest, graph->edge_data(node, ii)->rownext_edge);
    }
  }
  // Assign ordering into nodes
  for ( node_t ni = 0; ni < graph->nodecount; ni++ ) {
    graph->node_data(nodeorder[ni])->order = ni;
  }

  // Need traversal order to ensure update occur in correct order
  for ( node_t ni = 0; ni < graph->nodecount; ni++ ) {
    node_t node = nodeorder[ni];
    //printf("NODE %u\n", node);
    // Find edges between nodes, update rownext
    for ( edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
          ii != ei; ii++ ) {
      node_t src = graph->edge_dest(node, ii);
      for ( edge_t ij = graph->edge_begin(node), ej = graph->edge_end(node);
            ij != ej; ij++ ) {
        node_t dest = graph->edge_dest(node, ij);
        if ( src == dest || src == node )
          continue;
        edge_t bridge = graph->find_edge(src, dest);
        //printf("BRIDGE %u %u is edge %u\n", src, dest, bridge);
        if ( bridge != INVALID ) {
          edgedata_t *ijdata = graph->edge_data(node, ij);
          if ( graph->node_data(src)->order < graph->node_data(ijdata->rownext_node)->order ) {
            //printf("UPDATING %u %u rownext to be %u %u (%u)\n", node, dest, src, dest, bridge);
            ijdata->rownext_node = src;
            ijdata->rownext_edge = bridge;
          }
        }
      }
    }
  }
#endif
}

int main(int argc, char *argv[]) {
#ifdef ROWNAV
  if ( argc != 3 ) {
    fprintf(stderr, "Usage: %s <threadcount> <ordering>\n", argv[0]);
    return 1;
  }
#else
  if ( argc != 2 ) {
#ifdef SERIAL
    fprintf(stderr, "Usage: %s <ordering>\n", argv[0]);
#else
    fprintf(stderr, "Usage: %s <threadcount>\n", argv[0]);
#endif
    return 1;
  }
#endif

#if !defined(USE_HASH) && !defined(SERIAL)
  MutableGraph<true,true> graph;
#else
  MutableGraph<true> *temp = new MutableGraph<true>;
#ifdef SERIAL
  CRSGraph graph(*temp);
#else
  MapGraph graph(*temp);
#endif
  delete temp;
#endif
  compute_ebn(&graph, argv);

#ifdef SERIAL
  {
    node_t nodeorder[graph.nodecount];
    {
      FILE *fh = fopen(argv[1], "r");
      if ( !fh ) abort();
      unsigned x, i = 0;
      while ( fscanf(fh, "%u", &x) == 1 ) {
        assert(i < graph.nodecount);
        nodeorder[i] = x;
        i++;
      }
      fclose(fh);
      if ( i != graph.nodecount )
        abort();
    }
    printf("Using serial code...\n");
    struct timeval start, end;
    papi_start();
    gettimeofday(&start, NULL);
    for ( unsigned i = 0; i < graph.nodecount; i++ ) {
        do_node(&graph, nodeorder[i]);
    }
    gettimeofday(&end, NULL);
    papi_stop("");
    printf("1 threads done in %ld ms\n", TIMEDELTA_MICRO(start, end)/1000);
  }
#else
  int nthreads = atoi(argv[1]);
  run_dag(&graph, nthreads);
#endif

  //printf("counter=%u\n", counter);

  print_graph(&graph);
  return 0;
}
