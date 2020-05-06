/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

/* g++ -DNDEBUG leftcholesky.cpp -O3 -g -Wall -lpthread -std=c++11 */

#include <unistd.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <string.h>

#include "simplepapi.h"

typedef struct nodedata {
  int seen;
} nodedata_t;
typedef struct edgedata {
  double data;
  // pthread_mutex_t lock;
} edgedata_t;

#include "dag.h"

void init_node(node_t id, nodedata_t* data) { data->seen = 0; }

void init_edge(node_t from, node_t to, edgedata_t* data, char* extra) {
  /* Load edge weight (if data is non-NULL) */
  data->data = extra ? atof(extra) : 0; // 0 should never happen
}

// static unsigned counter = 0;

template <typename Graph>
void do_node(Graph* graph, node_t node) {
  {
    nodedata_t* nd = graph->node_data(node);
    // Update seen flag on node
    assert(nd->seen == 0);
    // Shouldn't need an atomic for this -- especially since it doesn't
    // depend on the previous value. Protected by DAG schedule (no more
    // incoming edges).
    nd->seen = 1;
  }

  // Find self-edge for this node
  edge_t selfedge = graph->find_edge(node, node);
  assert(selfedge != INVALID);
  double *selfdata_p = &graph->edge_data(node, selfedge)->data,
         selfdata    = *selfdata_p;
  // printf("STARTING %4d %10.5f\n", node, *selfdata_p);

  // Update self edge by L_ii=sqrt(L_ii-sum(L_ik*L_ik, k=0..i-1)) (sum
  // over incoming edges)
  for (edge_t ii = graph->inedge_begin(node), ei = graph->inedge_end(node);
       ii != ei; ii++) {
    node_t src = graph->inedge_src(node, ii);
    if (src == node || graph->node_data(src)->seen == 0)
      continue;
    double iidata = graph->inedge_data(node, ii)->data;
    iidata *= iidata;
    selfdata -= iidata;
    // printf(" L[%4d,%4d] -= L[%4d,%4d]^2 == %10.5f => %10.5f\n", node, node,
    // node, src, iidata, *selfdata_p);
  }
  assert(selfdata > 0);
  selfdata = sqrt(selfdata);
  assert(selfdata > 0 && !isnan(selfdata));
  *selfdata_p = selfdata;

  // Update all outgoing edges (matrix column) by
  // L_ji=(L_ji-sum(L_jk*L_ik, k=0..i-1))/L_ii (dot product incoming
  // edges to i and j)
  for (edge_t ci = graph->edge_begin(node), ce = graph->edge_end(node);
       ci != ce; ci++) {
    node_t dest = graph->edge_dest(node, ci);
    if (graph->node_data(dest)->seen != 0)
      continue;
    double *edgedata_p = &graph->edge_data(node, ci)->data,
           edgedata    = *edgedata_p;

    edge_t ii = graph->inedge_begin(node), ie = graph->inedge_end(node),
           ji = graph->inedge_begin(dest), je = graph->inedge_end(dest);
    while (ii < ie && ji < je) {
      node_t isrc = graph->inedge_src(node, ii),
             jsrc = graph->inedge_src(dest, ji);
      if (isrc == jsrc) {
        if (isrc != node && graph->node_data(isrc)->seen) {
          double delta = graph->inedge_data(node, ii)->data *
                         graph->inedge_data(dest, ji)->data;
          edgedata -= delta;
          // printf(" L[%4d,%4d] -= L[%4d,%4d]*L[%4d,%4d] == %10.5f =>
          // %10.5f\n", dest, node, node, isrc, dest, jsrc, delta, *edgedata_p);
        }

        // Increment row iterators. Note: inedges must be sorted!
        ii++;
        ji++;
      } else if (isrc < jsrc)
        ii++;
      else if (isrc > jsrc)
        ji++;
      else
        assert(false);
    }
    edgedata /= selfdata;
    *edgedata_p = edgedata;
    // printf(" L[%4d,%4d] /= L[%4d,%4d] == %10.5f => %10.5f\n", dest, node,
    // node, node, *selfdata_p, *edgedata_p);
  }
}

template <typename Graph>
void print_graph(Graph* graph) {
  /* Print all edges */
  FILE* fh = fopen("dagcholeskyedges.txt", "w");
  if (!fh)
    abort();
  for (node_t node = 0; node < graph->nodecount; node++) {
    for (edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
         ii != ei; ii++) {
      fprintf(fh, "%d %d %.*e\n", node, graph->edge_dest(node, ii), DBL_DIG + 3,
              graph->edge_data(node, ii)->data);
    }
  }
  if (fclose(fh) != 0)
    abort();
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
#ifdef SERIAL
    fprintf(stderr, "Usage: %s <ordering>\n", argv[0]);
#else
    fprintf(stderr, "Usage: %s <threadcount>\n", argv[0]);
#endif
    return 1;
  }

  MutableGraph<true>* temp = new MutableGraph<true>;
  BidiGraph graph(*temp);
  delete temp;

#ifdef SERIAL
  {
    node_t nodeorder[graph.nodecount];
    {
      FILE* fh = fopen(argv[1], "r");
      if (!fh)
        abort();
      unsigned x, i = 0;
      while (fscanf(fh, "%u", &x) == 1) {
        assert(i < graph.nodecount);
        nodeorder[i] = x;
        i++;
      }
      fclose(fh);
      if (i != graph.nodecount)
        abort();
    }
    printf("Using serial code...\n");

    struct timeval start, end;
    papi_start();
    gettimeofday(&start, NULL);
    for (unsigned i = 0; i < graph.nodecount; i++) {
      do_node(&graph, nodeorder[i]);
    }
    gettimeofday(&end, NULL);
    papi_stop("");
    printf("1 threads done in %ld ms\n", TIMEDELTA_MICRO(start, end) / 1000);
  }
#else
  int nthreads = atoi(argv[1]);
  run_dag(&graph, nthreads);
#endif

  // printf("counter=%u\n", counter);

  print_graph(&graph);
  return 0;
}
