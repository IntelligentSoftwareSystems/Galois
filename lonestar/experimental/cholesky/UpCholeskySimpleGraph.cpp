/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

//
// Fast up-looking Cholesky factorization
//
#include <unistd.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <string.h>

//#define PAPI
#include "simple/simplepapi.h"

#include "galois/Galois.h" // STRANGE ERRORS IF EXCLUDED
#include "galois/runtime/TreeExec.h"
#include "galois/Timer.h"

typedef struct nodedata {
  int seen;
} nodedata_t;
typedef struct edgedata {
  double data;
} edgedata_t;

#include "simple/dag.h"

void init_node(node_t id, nodedata_t* data) { data->seen = 0; }

void init_edge(node_t from, node_t to, edgedata_t* data, char* extra) {
  /* Load edge weight (if data is non-NULL) */
  data->data = extra ? atof(extra) : 0; // 0 should never happen
}

template <typename Graph>
struct PerThread {
  double* densecol;
  node_t* etree_stack;
  char* etree_mark;

  PerThread(Graph* graph) {
    // Initialize dense column for solve operation
    densecol = (double*)malloc(graph->nodecount * sizeof(double));
    if (!densecol)
      abort();
    memset(densecol, 0, graph->nodecount * sizeof(double));

    // Initialize elimination tree workspace
    etree_stack = (node_t*)malloc(graph->nodecount * sizeof(node_t));
    etree_mark  = (char*)malloc(graph->nodecount * sizeof(char));
    if (!etree_stack || !etree_mark)
      abort();
    memset(etree_stack, 0, graph->nodecount * sizeof(node_t));
    memset(etree_mark, 0, graph->nodecount * sizeof(char));
  }
};
typedef galois::substrate::PerThreadStorage<PerThread<CRSGraph>> GPTS;

node_t* etree        = NULL; // FIXME: ugly hack
CRSGraph* outgraph   = NULL; // not perthread
edge_t* outgraphptrs = NULL;

// Compute the elimination tree of A
template <typename Graph>
node_t* make_etree(const Graph* graph) {
  node_t* parent   = (node_t*)malloc(graph->nodecount * sizeof(node_t));
  node_t* ancestor = (node_t*)malloc(graph->nodecount * sizeof(node_t));
  if (!parent || !ancestor)
    abort();

  for (node_t n = 0; n < graph->nodecount; n++) {
    parent[n]   = INVALID;
    ancestor[n] = INVALID;
    for (edge_t ci = graph->edge_begin(n), ce = graph->edge_end(n); ci != ce;
         ci++) {
      node_t i = graph->edge_dest(n, ci);
      while (i < n && i != INVALID) {
        node_t nexti = ancestor[i];
        ancestor[i]  = n;
        if (nexti == INVALID)
          parent[i] = n;
        i = nexti;
      }
    }
  }

  free(ancestor);
  return parent;
}

/* find nonzero pattern of Cholesky L(k,1:k-1) using etree and triu(A(:,k)) */
template <typename Graph>
unsigned etree_reach(const Graph* graph, node_t k, const node_t* parent,
                     node_t* stack, char* marked) {
  unsigned stackpos = graph->nodecount;
  assert(!marked[k]);
  marked[k] = 1;

  for (edge_t ci = graph->edge_begin(k), ce = graph->edge_end(k); ci != ce;
       ci++) {
    node_t i = graph->edge_dest(k, ci);
    assert(i <= k);

    // Traverse up the elimination tree
    unsigned depth;
    for (depth = 0; !marked[i]; i = parent[i], depth++) {
      stack[depth] = i;
      // printf("Found [%u]: %u\n", depth, i);
      assert(marked[i] == 0);
      marked[i] = 1;
      // FIXME      assert(parent[i] < graph->size());
    }

    // Move traversed elements to the stack
    // FIXME: Why is this a separate step?
    // FIXME: Why is stack from the top, not the bottom?
    while (depth > 0) {
      stackpos--;
      depth--;
      stack[stackpos] = stack[depth];
      // printf("Found [%u] -> [%u]: %u == %u\n", depth, stackpos, stack[depth],
      // stack[stackpos]);
    }
  }

  // Unmark all nodes
  for (unsigned i = stackpos; i < graph->nodecount; i++)
    marked[stack[i]] = 0;
  marked[k] = 0;

#ifndef NDEBUG
  // FIXME:    for ( unsigned i = 0; i < graph->size(); i++ )
  // assert(marked[i] == 0);
#endif

  return stackpos;
}

template <typename Graph>
void do_node(PerThread<Graph>* mypts, Graph* graph, node_t node) {
  double* densecol = mypts->densecol;
  node_t* estack   = mypts->etree_stack;
  char* emark      = mypts->etree_mark;
  {
    nodedata_t* nd = graph->node_data(node);
    // Update seen flag on node
    assert(nd->seen == 0);
    nd->seen = 1;
  }
  // printf("STARTING %4d\n", node);

  // Use a dense row for solve operation
  edge_t selfedge = graph->edge_end(node) - 1; // ORDERING ASSUMPTION
  for (edge_t ci = graph->edge_begin(node),
              ce = selfedge /*graph->edge_end(node)*/;
       ci != ce; ci++) {
    node_t rownode = graph->edge_dest(node, ci);
    assert(rownode != node);
    assert(graph->node_data(rownode)->seen == 1);
    densecol[rownode] = graph->edge_data(node, ci)->data;
    // printf(" EXPLODE     %3u %f\n", rownode, densecol[rownode]);
  }
  // Find self-edge for this node
  assert(selfedge != INVALID);
  assert(graph->edge_dest(node, selfedge) == node);
  double *selfdata_p = &graph->edge_data(node, selfedge)->data,
         selfdata    = *selfdata_p;

  // Find non-zero pattern of row
  unsigned stackpos = etree_reach(graph, node, etree, estack, emark);
  // printf("stack: %u\n", graph->nodecount-stackpos);

  // Solve for outgoing edges (matrix row)
  // L[k,1:k-1]=L[1:k-1,1:k-1]\A[1:k-1,k]
  for (; stackpos < graph->nodecount; stackpos++) {
    node_t colnode = estack[stackpos];
    // printf(" COL %4u\n", colnode);
    assert(colnode != node);                      // ORDERING ASSUMPTION
    assert(graph->node_data(colnode)->seen == 1); // ORDERING ASSUMPTION
    double x = densecol[colnode]; // Right-hand side of this linear equation
    densecol[colnode] = 0;

    // Find diagonal entry
    edge_t coldiag_edge =
        graph->edge_end(colnode) -
        1; // graph->graph->node_data(colnode)->diag; // ORDERING ASSUMPTION
    // printf("%d %d: %d %d\n", colnode, coldiag_edge, graph->edge_dest(colnode,
    // coldiag_edge), colnode);
    assert(graph->edge_dest(colnode, coldiag_edge) == colnode);

    {
      // Update entry in column
      double coldiag = graph->edge_data(colnode, coldiag_edge)->data;
      assert(coldiag != 0);
      // densecol[colnode] = 0;
      // printf("  L[%4d,%4d] == %10.5f\n", colnode, node, x);
      x /= coldiag; // Divide by diagonal entry
      // printf("  L[%4d,%4d] /= L[%4d,%4d] == %10.5f => %10.5f\n", colnode,
      // node, colnode, colnode, coldiag, x);
    }

    {
      // FIXME: Assume ordering works for triangular solve. DAG schedule?
      // printf("  L[%4d,%4d] == %10.5f\n", node, node, selfdata);
      selfdata -= x * x;
      // printf("  L[%4d,%4d] -= (x == %10.5f)^2 => %10.5f\n", node, node, x,
      // selfdata);
    }

    edge_t ji           = outgraph->edge_begin(colnode) + 1;
    node_t* jdest_p     = &(outgraph->edgedest[ji]);
    edgedata_t* jdata_p = &(outgraph->edgedata[ji]);
    for (edge_t je = outgraphptrs[colnode]; ji < je; ji++) {
      node_t jdest = *(jdest_p++);
      double jdata = (jdata_p++)->data;
      // assert(graph->node_data(jdest)->seen == 1); // ORDERING ASSUMPTION
      double delta = x * jdata;
      // printf("  col[%4d] = %10.5f - (%10.5f * %10.5f == %10.5f)\n", jdest,
      // densecol[jdest], x, jdata, delta);
      densecol[jdest] -= delta;
      // printf("  => %10.5f\n", densecol[jdest]);

      // printf("  L[%4d,%4d] == %10.5f, L[%4d,%4d] == %10.5f\n", node, jdest,
      // densecol[jdest], colnode, jdest, jdata); printf("  b -= (%10.5f * %10.5f
      // == %10.5f) => %10.5f\n", densecol[jdest], jdata, delta, b);
      ////printf(" L[%4d,%4d] -= L[%4d,%4d]*L[%4d,%4d] == %10.5f => %10.5f\n",
      ///node, colnode, node, idest, colnode, jdest, delta, b);
      // printf("  %u\n", jdest);
    }
    {
      // printf("Writing %u -> %u == %f\n", colnode, node, x);
      // edge_t q = outgraphptrs[colnode]++; // ATOMIC
      edge_t q = __sync_fetch_and_add(&outgraphptrs[colnode], 1);
      assert(q < outgraph->edgeidx[node]);
      outgraph->edgedest[q]      = node;
      outgraph->edgedata[q].data = x;
    }
  }

  // Update self edge by L[i,i]=sqrt(L[i,i]-sum(L[k,i]^2, k=0..i-1))
  // (sum over outgoing edges/matrix row)
  assert(selfdata > 0 && !isnan(selfdata));
  selfdata    = sqrt(selfdata);
  *selfdata_p = selfdata; // FIXME: Remove this, but avoid perf penalty
  {
    // printf("Writing %u -> %u == %f\n", node, node, selfdata);
    // edge_t q = outgraphptrs[node]++; // ATOMIC
    edge_t q = __sync_fetch_and_add(&outgraphptrs[node], 1);
    assert(q < outgraph->edgeidx[node]);
    outgraph->edgedest[q]      = node;
    outgraph->edgedata[q].data = selfdata;
  }
  // printf(" L[%4d,%4d] = sqrt(L[%4d,%4d] == %10.5f) => %10.5f\n", node, node,
  // node, node, selfdata, *selfdata_p);
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
      node_t dest = graph->edge_dest(node, ii);
      fprintf(fh, "%d %d %.*e\n", node, dest, DBL_DIG + 3,
              graph->edge_data(node, ii)->data);
    }
  }
  if (fclose(fh) != 0)
    abort();
}

static void reverse_graph(CRSGraph* graph) {
  // Reverse a CRSGraph in-place using a temporary BidiGraph
  BidiGraph temp(*graph);
  unsigned i;
  unsigned edgecount = graph->edgeidx[graph->nodecount - 1];
  memcpy(graph->edgeidx, temp.inedgeidx, graph->nodecount * sizeof(edge_t));
  memcpy(graph->edgedest, temp.inedgesrc, edgecount * sizeof(node_t));
  for (i = 0; i < edgecount; i++) {
    graph->edgedata[i] = temp.edgedata[temp.inedgedataidx[i]];
  }
}

template <typename Graph>
struct TreeExecModel {
  typedef std::vector<node_t> ChildList;
  typedef std::vector<ChildList> Children;
  Children children;
  ChildList rootnodes;
  Graph* graph;
  GPTS* pts;

  TreeExecModel(Graph* graph, GPTS* pts)
      : graph(graph), pts(pts), children(graph->nodecount, ChildList()) {
    for (node_t i = 0; i < graph->nodecount; i++) {
      if (etree[i] == INVALID)
        rootnodes.push_back(i);
      else
        children[etree[i]].push_back(i);
    }
  }

  struct GaloisDivide {
    TreeExecModel* tem;
    int rootnodes;
    GaloisDivide(TreeExecModel* tem) : tem(tem), rootnodes(1) {}
    template <typename C>
    void operator()(node_t i, C& ctx) {
      unsigned j;
      for (j = 0; j < tem->children[i].size(); j++) {
        ctx.spawn(tem->children[i][j]);
      }
      if (__sync_bool_compare_and_swap(&rootnodes, 1, 0)) {
        for (j = 1; j < tem->rootnodes.size(); j++) {
          ctx.spawn(tem->rootnodes[j]);
        }
      }
    }
  };

  struct GaloisConquer {
    TreeExecModel* tem;
    GaloisConquer(TreeExecModel* tem) : tem(tem) {}
    void operator()(unsigned int i) {
      do_node(tem->pts->getLocal(), tem->graph, i);
    };
  };

  void run() {
    // Begin execution -- FIXME: initial elements
    printf("%u root nodes\n", rootnodes.size());
    galois::runtime::for_each_ordered_tree(rootnodes[0], GaloisDivide(this),
                                           GaloisConquer(this),
                                           "UpCholeskyTree");
  }
};

int main(int argc, char* argv[]) {
  galois::StatManager statManager;
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <threadcount>\n", argv[0]);
    return 1;
  }

  MutableGraph<true>* temp = new MutableGraph<true>;
  CRSGraph graph(*temp), theoutgraph(*temp);
  delete temp;
  reverse_graph(&graph);
  etree    = make_etree(&graph);
  outgraph = &theoutgraph;

  {
    // Starting points for insertions into outgraph
    outgraphptrs = (edge_t*)malloc(outgraph->nodecount * sizeof(edge_t));
    if (!outgraphptrs)
      abort();
    outgraphptrs[0] = 0;
    for (unsigned i = 1; i < outgraph->nodecount; i++) {
      outgraphptrs[i] = outgraph->edgeidx[i - 1];
    }
  }

  int nthreads = atoi(argv[1]);
  nthreads     = galois::setActiveThreads(nthreads);

  GPTS pts(&graph);
  galois::StatTimer T("NumericTime");
  struct timeval start, end;
  papi_start();
  T.start();
  gettimeofday(&start, NULL);
  TreeExecModel<CRSGraph>(&graph, &pts).run();
  gettimeofday(&end, NULL);
  T.stop();
  papi_stop("");
  printf("%d threads done in %ld ms\n", nthreads,
         TIMEDELTA_MICRO(start, end) / 1000);
  print_graph(&theoutgraph);
  return 0;
}
