/* g++ -DNDEBUG leftcholesky.cpp -O3 -g -Wall -lpthread -std=c++11 */

#include <unistd.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <string.h>

#include "simple/simplepapi.h"

#include "galois/Galois.h"
#include "galois/runtime/TreeExec.h"
#include "galois/Timer.h"

#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#define MI(i,j,istride,jstride) ((i)*(istride)+(j)*(jstride))
extern "C" {
#include <cblas.h>
#include <clapack.h>
}

typedef struct nodedata {} nodedata_t;
typedef struct edgedata {
  double data;
  //pthread_mutex_t lock;
} edgedata_t;

#include "simple/dag.h"

void init_node(node_t id, nodedata_t *data) {
}

void init_edge(node_t from, node_t to, edgedata_t *data, char *extra) {
  /* Load edge weight (if data is non-NULL) */
  data->data = extra ? atof(extra) : 0; // 0 should never happen
}


struct SuperGraph {
  unsigned *nodesize;           // Size of each supernode
  unsigned nodecount;           // Number of supernodes

  // End iterator for super-edges of each supernode (size: nodecount)
  edge_t *edgeidx;
  // Destination for each super-edge (size: nedges_est)
  node_t *edgedest;
  // Index in data of edge submatrix (size: nedges_est)
  unsigned *indirect;
  // Data submatrices (size: m_datasize)
  double *data;

  // Edge Rows
  unsigned *edgerowidx;         // Indexing into edgerows for each super-edge
  unsigned *edgerows;           // List of rows for each super-edge

  unsigned edge_rowstart(node_t node, edge_t edge) {
    return edge > 0 ? edgerowidx[edge-1] : 0;
  }
  unsigned edge_rowend(node_t node, edge_t edge) {
    return edgerowidx[edge];
  }
  unsigned edge_rowcount(node_t node, edge_t edge) {
    return edge_rowend(node, edge)-edge_rowstart(node, edge);
  }

  edge_t edge_begin(node_t node) const {
    return node > 0 ? edgeidx[node-1] : 0;
  }
  edge_t edge_end(node_t node) const {
    return edgeidx[node];
  }
  double *edge_data(node_t node, edge_t edge) {
    return &data[indirect[edge]];
  }
  node_t edge_dest(node_t node, edge_t edge) const {
    return edgedest[edge];
  }

  // Reverse matrix
  edge_t *inedgeidx;
  node_t *inedgesrc;
  unsigned *inedgeindirect;
  unsigned *inedgerowidxindirect; // EXTRA INDIRECTION - FIXME

  void make_inedges() {
    unsigned edgecount = edgeidx[nodecount-1];
    inedgeidx = (edge_t *)malloc(nodecount * sizeof(edge_t));
    inedgesrc = (node_t *)malloc(edgecount * sizeof(node_t));
    inedgeindirect = (unsigned *)malloc(edgecount * sizeof(unsigned));
    inedgerowidxindirect = (unsigned *)malloc(edgecount * sizeof(unsigned));
    unsigned temp[nodecount];
    // Count incoming edges for each node
    for ( unsigned i = 0; i < nodecount; i++ )
      temp[i] = 0;
    for ( node_t src = 0; src < nodecount; src++ ) {
      for ( edge_t ii = edge_begin(src), ei = edge_end(src); ii != ei; ii++ ) {
        node_t dest = edge_dest(src, ii);
        temp[dest]++;
      }
    }
    // Store cumulative sums as inedgeidx (compressed storage)
    for ( unsigned i = 0; i < nodecount; i++ ) {
      unsigned start = i > 0 ? inedgeidx[i-1] : 0;
      inedgeidx[i] = temp[i] + start;
      temp[i] = start;
    }
    // Insert edges into list
    for ( node_t src = 0; src < nodecount; src++ ) {
      for ( edge_t ii = edge_begin(src), ei = edge_end(src); ii != ei; ii++ ) {
        node_t dest = edge_dest(src, ii);
        inedgesrc[temp[dest]] = src;
        inedgeindirect[temp[dest]] = indirect[ii];
        inedgerowidxindirect[temp[dest]] = ii;
        temp[dest]++;
      }
    }
    // Sort incoming edges (linear time sort using temp)
    for ( unsigned i = 0; i < nodecount; i++ )
      temp[i] = INVALID;
    for ( node_t dest = 0; dest < nodecount; dest++ ) {
      // Explode inedges data into temp
      for ( edge_t ii = inedge_begin(dest), ei = inedge_end(dest);
            ii != ei; ii++ ) {
        temp[inedge_src(dest, ii)] = inedgeindirect[ii];
      }
      // Recompress inedges in sorted order
      unsigned idx = inedge_begin(dest);
      for ( unsigned j = 0; j < nodecount; j++ ) {
        if ( temp[j] != INVALID ) {
          inedgesrc[idx] = j;
          inedgeindirect[idx] = temp[j];
          temp[j] = INVALID;
          idx++;
        }
      }
      assert(idx == inedge_end(dest));
    }
  }

  edge_t inedge_begin(node_t node) const {
    return node > 0 ? inedgeidx[node-1] : 0;
  }

  edge_t inedge_end(node_t node) const {
    return inedgeidx[node];
  }

  double *inedge_data(node_t node, edge_t edge) {
    return &data[inedgeindirect[edge]];
  }
  node_t inedge_src(node_t node, edge_t edge) const {
    return inedgesrc[edge];
  }

  // rowcount funcs
  unsigned inedge_rowstart(node_t node, edge_t edge) {
    edge_t outedge = inedgerowidxindirect[edge];
    return edge_rowstart(INVALID, outedge);
  }
  unsigned inedge_rowend(node_t node, edge_t edge) {
    edge_t outedge = inedgerowidxindirect[edge];
    return edge_rowstart(INVALID, outedge);
  }
  unsigned inedge_rowcount(node_t node, edge_t edge) {
    edge_t outedge = inedgerowidxindirect[edge];
    return edge_rowcount(INVALID, outedge);
  }

};

void do_node(SuperGraph *graph, node_t node, double *tempmatrix) {
  // Find self-edge for this node
  edge_t selfedge = graph->edge_begin(node);
  assert(graph->edge_dest(node, selfedge) == node);
  assert(selfedge != INVALID);
  double *selfdata = graph->edge_data(node, selfedge);
  unsigned nodesize_node = graph->nodesize[node];
  //printf("STARTING %4d %10.5f\n", node, *selfdata_p);

  // FIXME: Consider replacing inedges with separate (transposed)
  // output matrix. This will allow use of rolling pointers instead of
  // repeated indirection (which might be faster).

  // Update self edge by L_ii=sqrt(L_ii-sum(L_ik*L_ik, k=0..i-1)) (sum
  // over incoming edges)
  for ( edge_t ii = graph->inedge_begin(node), ei = graph->inedge_end(node)-1;
        ii != ei; ii++ ) {
    node_t src = graph->inedge_src(node, ii);
    assert(src < node);
    double* iidata = graph->inedge_data(node, ii);
    // Subtract outer product of iidata from selfdata
    unsigned nodesize_src = graph->nodesize[src];
    unsigned *edgerows = &graph->edgerows[graph->inedge_rowstart(node, ii)];
    /*
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                nodesize_node, nodesize_node,
                nodesize_src, -1.0, iidata, nodesize_node,
                iidata, nodesize_node, 1.0, selfdata,
                nodesize_node);
    */
    /*
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                nodesize_node, nodesize_src, -1.0, iidata, nodesize_node,
                1.0, selfdata, nodesize_node);
    */
    // Use temporary C due to reduced number of rows
    unsigned edgerow_count = graph->inedge_rowcount(node, ii);
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                edgerow_count, nodesize_src, 1.0, iidata, edgerow_count,
                0, tempmatrix, edgerow_count);
    // Migrate data to selfdata
    unsigned iindex = 0;
    unsigned *edgerows_ptrj = edgerows;
    for ( unsigned mj = 0; mj < edgerow_count; mj++ ) {    // Over columns
      unsigned destcol = *(edgerows_ptrj++);
      assert(destcol == edgerows[mj]);
      unsigned oindex = destcol*nodesize_node;
      iindex += mj;
      unsigned *edgerows_ptri = edgerows + mj;
      // Optimization: only lower-triangular part
      for ( unsigned mi = mj; mi < edgerow_count; mi++ ) { // Over rows
        unsigned destrow = *(edgerows_ptri++);
        assert(destrow == edgerows[mi]);
        assert(iindex == MI(mi, mj, 1, edgerow_count));
        assert(oindex == MI(destrow, destcol, 1, nodesize_node));
        selfdata[oindex+destrow] -= tempmatrix[iindex];
        iindex++;
      }
    }
    //printf(" L[%4d,%4d] -= L[%4d,%4d]^2 == %10.5f => %10.5f\n", node, node, node, src, iidata, *selfdata_p);
  }
  int info = clapack_dpotrf(CblasColMajor, CblasLower, nodesize_node,
                            selfdata, nodesize_node);
  
  if ( info != 0 ) {
    fprintf(stderr, "dpotrf: returned %d\n", info);
    abort();
  }

  // Update all outgoing edges (matrix column) by
  // L_ji=(L_ji-sum(L_jk*L_ik, k=0..i-1))/L_ii (dot product incoming
  // edges to i and j)
  for ( edge_t ci = graph->edge_begin(node)+1, ce = graph->edge_end(node);
        ci != ce; ci++ ) {
    node_t dest = graph->edge_dest(node, ci);
    assert(dest > node);
    double *edgedata = graph->edge_data(node, ci);
    unsigned edgerow_count = graph->edge_rowcount(node, ci);
    unsigned *edgerows = &graph->edgerows[graph->edge_rowstart(node, ci)];

    edge_t ii = graph->inedge_begin(node), ie = graph->inedge_end(node)-1,
      ji = graph->inedge_begin(dest), je = graph->inedge_end(dest)-1;
    while ( ii < ie && ji < je ) {
      node_t isrc = graph->inedge_src(node, ii),
        jsrc = graph->inedge_src(dest, ji);
      if ( isrc == jsrc ) {
        assert(isrc < node);
        double *a = graph->inedge_data(node, ii);
        double *b = graph->inedge_data(dest, ji);
        // FIXME: Wrap all this in methods
        unsigned arow_count = graph->inedge_rowcount(node, ii);
        unsigned brow_count = graph->inedge_rowcount(dest, ji);
        unsigned *aedgerows = &graph->edgerows[graph->inedge_rowstart(node, ii)];
        unsigned *bedgerows = &graph->edgerows[graph->inedge_rowstart(dest, ji)];
        // Note that since matrix multiplication is noncommutative,
        // edgedata -= a*b becamse edgedata -= b*a;
        unsigned nodesize_src = graph->nodesize[isrc/*==jsrc*/];
        /*
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nodesize_dest, nodesize_node,
                    nodesize_src, -1.0, b,
                    nodesize_dest, a, nodesize_node, 1.0,
                    edgedata, nodesize_dest);
        */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    /*m*/brow_count, /*n*/arow_count,
                    /*k=cols*/nodesize_src, 1.0, b,
                    brow_count, a, arow_count, 0,
                    tempmatrix, brow_count);

        // Migrate data to edgedata
        unsigned iindex = 0;
        unsigned *aedgerows_ptr = aedgerows;
        for ( unsigned mj = 0; mj < arow_count; mj++ ) {    // Over columns
          unsigned destcol = *(aedgerows_ptr++);
          assert(destcol == aedgerows[mi]);
          assert(destcol >= 0 && destcol < nodesize_node);
          unsigned oindex = destcol*edgerow_count;
          unsigned destrow = 0;
          unsigned *bedgerows_ptr = bedgerows;
          unsigned *edgerows_ptr = edgerows;
          for ( unsigned mi = 0; mi < brow_count; mi++ ) { // Over rows
            unsigned brow = *(bedgerows_ptr++), erow = *(edgerows_ptr++);
            assert(brow == bedgerows[mi]);
            assert(erow == edgerows[destrow]);
            while ( brow > erow ) { // Skip this erow
              destrow++, oindex++;
              erow = *(edgerows_ptr++);
              assert(erow == edgerows[destrow]);
            }
            //if ( brow < erow ) { abort(); mi++; continue; }
            assert(brow == erow);
            assert(destrow >= 0 && destrow < edgerow_count);

            assert(iindex == MI(mi, mj, 1, brow_count));
            assert(oindex == MI(destrow, destcol, 1, edgerow_count));
            edgedata[oindex] -= tempmatrix[iindex];
            destrow++, oindex++;
            iindex++;
          }
        }
        //printf(" L[%4d,%4d] -= L[%4d,%4d]*L[%4d,%4d] == %10.5f => %10.5f\n", dest, node, node, isrc, dest, jsrc, delta, *edgedata_p);

        // Increment row iterators. Note: inedges must be sorted!
        ii++;
        ji++;
      }
      else if ( isrc < jsrc )
        ii++;
      else if ( isrc > jsrc )
        ji++;
      else
        assert(false);
    }
    //edgedata /= selfdata;
    // Becomes solve operation x*selfdata=edgedata' (edgedata <- x)
    /*
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                CblasNonUnit, nodesize_dest, nodesize_node,
                1.0, selfdata, nodesize_node, edgedata,
                nodesize_dest);
    */
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                CblasNonUnit, /*m*/edgerow_count, /*n*/nodesize_node,
                1.0, selfdata, /*==n*/nodesize_node, edgedata,
                /*==m*/edgerow_count);
    //printf(" L[%4d,%4d] /= L[%4d,%4d] == %10.5f => %10.5f\n", dest, node, node, node, *selfdata_p, *edgedata_p);
  }
}

node_t *etree = NULL;           // FIXME: ugly hack
// Compute the elimination tree of A
template <typename Graph>
node_t *make_etree(const Graph *graph) {
  node_t *parent = (node_t *)malloc(graph->nodecount*sizeof(node_t));
  node_t *ancestor = (node_t *)malloc(graph->nodecount*sizeof(node_t));
  if ( !parent || !ancestor )
    abort();

  for ( node_t n = 0; n < graph->nodecount; n++ ) {
    parent[n] = INVALID;
    ancestor[n] = INVALID;
    for ( edge_t ci = graph->inedge_begin(n), ce = graph->inedge_end(n);
          ci != ce; ci++ ) {
      node_t i = graph->inedge_src(n, ci);
      while ( i < n && i != INVALID ) {
        node_t nexti = ancestor[i];
        ancestor[i] = n;
        if ( nexti == INVALID )
          parent[i] = n;
        i = nexti;
      }
    }
  }

  free(ancestor);
  return parent;
}

struct mapping { unsigned n, p; };
template <typename Graph>
inline void make_supergraph(Graph *graph, SuperGraph *sg,
                     unsigned *max_c_size_ptr, struct mapping **mapping_ptr) {
  // graph->nodecount indicates the size of the supernode represented
  // by each node of the graph. Nodes that are part of a supernode,
  // but not the representative node, have size 0.
  unsigned *supersize = (unsigned *)malloc(graph->nodecount*sizeof(unsigned));
  if ( !supersize ) abort();
  sg->nodecount = 0; // Number of new nodes (supernodes+remaining)

  // Compute supernode sizes
  {
    // Compute number of children of each node
    unsigned *children = (unsigned *)malloc(graph->nodecount*sizeof(unsigned));
    if ( !children ) { perror("malloc children"); abort(); }
    memset(children, 0, graph->nodecount*sizeof(unsigned));
    for ( unsigned i = 0; i < graph->nodecount; i++ ) {
      unsigned p = etree[i];    // parent node
      if ( p == INVALID ) continue;
      children[p]++;
    }

    // Initialize supernode sizes to one (all "supernodes" have one node)
    for ( unsigned i = 0; i < graph->nodecount; i++ )
      supersize[i] = 1;

    // Consolidate nodes into supernodes
    for ( unsigned i = 0; i < graph->nodecount; i++ ) {
      if ( supersize[i] != 1 ) continue;
      unsigned n = i;
      unsigned count_n = graph->edge_end(n) - graph->edge_begin(n);
      unsigned supercount = 1;
      while ( 1 ) {
        unsigned p = etree[n];  // parent node
        // Check conditions for parent to be part of supernode
        if ( p == INVALID ) break;
        if ( children[p] != 1 ) break;
        unsigned count_p = graph->edge_end(p) - graph->edge_begin(p);
        assert(count_n-1 <= count_p);
        if ( count_n-1 != count_p ) break;
        //if ( count_p-(count_n-1) > 5 ) break
        // Add parent to supernode
        supercount++;
        supersize[p] = 0;
        // Move up the tree
        n = p;
        count_n = count_p;
      }
      supersize[i] = supercount;
    }
    free(children);
  }

  // Display count of supernodes
  {
    // supersize now has the number of nodes to include in each supernode
    unsigned totalsupers = 0, totala = 0, totall = 0;
    for ( unsigned i = 0; i < graph->nodecount; i++ ) {
      if ( supersize[i] > 1 ) {
        totalsupers++;
        totala += supersize[i];
      }
      else if ( supersize[i] == 1 ) totall++;
      if ( supersize[i] != 0 ) sg->nodecount++; // # supernodes + # remaining nodes
    }
    assert(sg->nodecount <= graph->nodecount-totalsupers);
    assert(totall == graph->nodecount-totala);
    assert(sg->nodecount == totalsupers+totall);
    printf("Number of supernodes: %d, comprising %d nodes (%d new tasks, L=%d)\n",
           totalsupers, totala, sg->nodecount, totall);
  }

  // Rewrite matrix and elimination tree

  // Create mapping from old node to supernode + position in supernode
  struct mapping *mapping = (struct mapping *)malloc(graph->nodecount*sizeof(struct mapping));
  *mapping_ptr = mapping;       // Reversed data is used for graph output

  // Fill supergraph with size of each supernode
  sg->nodesize = (unsigned *)malloc(sg->nodecount*sizeof(unsigned));
  for ( unsigned k = 0, i = 0; i < graph->nodecount; i++ ) {
    unsigned n = i;
    if ( supersize[n] == 0 ) continue;
    sg->nodesize[k] = supersize[n];
    // Walk up elimination tree, updating each position
    for ( unsigned j = 0; j < supersize[i]; j++ ) {
      assert(n == i || supersize[n] == 0);
      mapping[n].n = k;
      mapping[n].p = j;
      //printf("%d => %d, %d\n", n, k, j);
      n = etree[n];
    }
    k++;
  }

  // Build graph of supernodes.
  // FIXME: Might include assumption that edges are ordered?

  {
    // Estimate number of edges in the supergraph
    unsigned nedges_orig = graph->edge_end(graph->nodecount-1);
    unsigned nedges_est = nedges_orig; // FIXME: very large estimate

    // End iterator for super-edges of each supernode
    sg->edgeidx = (edge_t *)malloc(sg->nodecount*sizeof(edge_t));
    if ( !sg->edgeidx ) abort();
    // Destination for each super-edge
    sg->edgedest = (node_t *)malloc(nedges_est*sizeof(node_t));
    if ( !sg->edgedest ) abort();
  }

  // Construct structure of supergraph
  for ( unsigned n = 0, i = 0; i < sg->nodecount; i++ ) {
    unsigned a = i > 0 ? sg->edgeidx[i-1] : 0, b = a; // Range for this node
    assert(mapping[n].n == i);
    // Neighbors of this node are superset of neighbors of parent nodes.
    for ( edge_t ii = graph->edge_begin(n), ei = graph->edge_end(n);
          ii != ei; ii++ ) {
      unsigned d = graph->edge_dest(n, ii);
      unsigned ds = mapping[d].n;
      unsigned found = 0;
      for ( unsigned j = b; j > a; j-- ) {
        // Check if neighbor reused
        if ( sg->edgedest[j-1] == ds ) {
          found = 1;
          break;
        }
      }
      if ( !found ) {
        // Add new edge between supernodes
        sg->edgedest[b] = ds;
        b++;
      }
    }
    sg->edgeidx[i] = b;
    n++;
    while ( supersize[n] == 0 ) n++; // Supernodes might not be contiguous
  }

  // Pointers to data submatrix for each super-edge
  unsigned nsuperedges = sg->edgeidx[sg->nodecount-1];
  sg->indirect = (unsigned *)malloc(nsuperedges*sizeof(unsigned));
  if ( !sg->indirect ) abort();

  // Row indexes for each super-edge
  sg->edgerowidx = (node_t *)calloc(nsuperedges, sizeof(unsigned));
  if ( !sg->edgerowidx ) abort();

  // Compute rows of each super-edge
  for ( unsigned n = 0, i = 0; i < sg->nodecount; i++ ) {
    // Range for this node
    unsigned a = i > 0 ? sg->edgeidx[i-1] : 0, b = sg->edgeidx[i];
    assert(mapping[n].n == i);
    // Neighbors of this node are superset of neighbors of parent nodes.
    for ( edge_t ii = graph->edge_begin(n), ei = graph->edge_end(n);
          ii != ei; ii++ ) {
      unsigned d = graph->edge_dest(n, ii);
      unsigned ds = mapping[d].n;
      unsigned found = 0;

      for ( unsigned j = b; j > a; j-- ) {
        // Check if neighbor reused
        if ( sg->edgedest[j-1] == ds ) {
          found = 1;
          // Add to rowcount for this super-edge
          sg->edgerowidx[j-1]++;
          break;
        }
      }
      assert(found);
    }
    n++;
    while ( supersize[n] == 0 ) n++; // Supernodes might not be contiguous
  }

  // Partial sums of edgerowidx, giving ranges in edgerows
  for ( unsigned j = 1; j < nsuperedges; j++ )
    sg->edgerowidx[j] += sg->edgerowidx[j-1];

  // Temp pointers into edgerows for incremental storage
  unsigned *edgerowtemp = (node_t *)calloc(nsuperedges, sizeof(unsigned));
  if ( !edgerowtemp ) abort();

  {
    // Compute size of each super-edge
    unsigned m_datasize = 0;
    unsigned edgerows_size = 0;
    unsigned max_c_size = 0;
    for ( unsigned i = 0; i < sg->nodecount; i++ ) {
      for ( unsigned j = (i > 0) ? sg->edgeidx[i-1] : 0;
            j < sg->edgeidx[i]; j++ ) {
        sg->indirect[j] = m_datasize;
        unsigned rowcount = sg->edgerowidx[j]-((j>0) ? sg->edgerowidx[j-1] : 0);
        unsigned cellsize = sg->nodesize[i]*rowcount;
        if ( cellsize > max_c_size ) max_c_size = cellsize;
        m_datasize += cellsize;
        edgerows_size += rowcount;
      }
    }
    *max_c_size_ptr = max_c_size; // Used for per-thread storage, do_node

    assert(edgerows_size == sg->edgerowidx[sg->edgeidx[sg->nodecount-1]-1]);
    // FIXME: duplication of data?

    // Allocate edge-row listing (each row for each super-edge)
    sg->edgerows = (unsigned *)malloc(edgerows_size*sizeof(unsigned));
    if ( !sg->edgerows ) abort();

    // Allocate Data matrix
    sg->data = (double *)calloc(m_datasize, sizeof(double));
    if ( !sg->data ) abort();

    // Display difference in nonzero count
    {
      unsigned actualnonzeros = m_datasize;
      for ( unsigned i = 0; i < sg->nodecount; i++ ) {
        unsigned wasteddiag = (sg->nodesize[i]-1)*sg->nodesize[i]/2;
        actualnonzeros -= wasteddiag;
      }
      printf("Expected nonzeros: %u; Actual nonzeros: %u\n",
             graph->edge_end(graph->nodecount-1), actualnonzeros);
    }
  }

  {
    // Now organize the data - Find edge between given supernodes
    // (nontrivial) and write correct submatrix entry. Actually, can
    // do similar condensation on edge list.
    unsigned oldsuperi = INVALID;
    unsigned si, ii, ei;
    for ( unsigned i = 0; i < graph->nodecount; i++ ) {
      // Edge from node i goes from supernode superi, submatrix position posi.
      unsigned superi = mapping[i].n, posi = mapping[i].p;

      // Keep a persistent pointer (ii) in the super-edge list, for
      // fast location of corresponding super-edge
      if ( superi != oldsuperi ) {
        ii = si = superi > 0 ? sg->edgeidx[superi-1] : 0;
        ei = sg->edgeidx[superi];
        oldsuperi = superi;
      }

      // Enumerate all edges from this node in the original graph
      for ( edge_t ij = graph->edge_begin(i), ej = graph->edge_end(i);
          ij != ej; ij++ ) {
        // Edge to node j goes to supernode superj, submatrix position posj.
        unsigned j = graph->edge_dest(i, ij);
        unsigned superj = mapping[j].n, rowj = mapping[j].p;

        // Find edge between supernodes. If we keep a persistent
        // pointer we shouldn't have to move it too far.
        while ( sg->edgedest[ii] != superj ) {
          ii++;
          if ( ii >= ei ) ii = si;
        }

        // Find posj
        unsigned edgerows_start = (ii > 0) ? sg->edgerowidx[ii-1] : 0;
        unsigned maxposj = sg->edgerowidx[ii]-edgerows_start;
        unsigned posj = edgerowtemp[ii];
        unsigned edgerows_idx = edgerows_start+posj;

        // Search edgerows for posj
        unsigned found = 0;
        for ( ; edgerows_idx > edgerows_start; edgerows_idx--, posj-- ) {
          // Check if neighbor reused
          if ( sg->edgerows[edgerows_idx-1] == rowj ) {
            found = 1;
            edgerows_idx--;
            posj--;
            break;
          }
        }
        if ( !found ) {
          // Add to edgerows
          posj = edgerowtemp[ii];
          edgerowtemp[ii]++;
          edgerows_idx = edgerows_start+posj;
          sg->edgerows[edgerows_idx] = mapping[j].p;
        }

        // Verify posj
        //printf("%d -> %d going to superedge %d -> %d at (%d,%d)\n",
        //       i, j, superi, superj, posi, posj);
        assert(edgerows_idx < sg->edgerowidx[ii]);
        assert(edgerows_idx <= edgerows_start ||
               sg->edgerows[edgerows_idx] > sg->edgerows[edgerows_idx-1] /* out-of-order */);
        assert(maxposj <= sg->nodesize[superj]);

        // Insert data into the edge. (FIXME: slow?)
        double *matrix = &sg->data[sg->indirect[ii]];
        unsigned idx = MI(posi, posj, maxposj, 1);
        assert(idx < sg->nodesize[superi]*maxposj);
        matrix[idx] = graph->edge_data(i, ij)->data;
        /*
        printf("ij=(%d,%d), superij=(%d,%d), posij=(%d,%d)%%%u => %u => %f\n",
               i, j, superi, superj, posi, posj, sg->nodesize[superi], idx,
               graph->edge_data(i, ij)->data);
        */
      }
    }
  }

  free(edgerowtemp);

  // Create reversed graph for traversal of incoming edges
  sg->make_inedges();
}

void print_supergraph(SuperGraph *graph,
                      unsigned *revmappingidx, unsigned *revmapping) {
  /* Print all edges */
  FILE *fh = fopen("dagcholeskyedges.txt", "w");
  if ( !fh ) abort();
  unsigned *rmn = revmapping;
  for ( node_t node = 0; node < graph->nodecount; node++ ) {
    for ( edge_t ii = graph->edge_begin(node), ei = graph->edge_end(node);
          ii != ei; ii++ ) {
      unsigned dest = graph->edge_dest(node, ii);
      //fprintf(fh, "SN: %d %d\n", node, dest);
      unsigned *rmd = &revmapping[dest > 0 ? revmappingidx[dest-1] : 0];
      double *matrix = graph->edge_data(node, ii);
      unsigned maxj = graph->edge_rowcount(node, ii);
      unsigned *edgerows = &graph->edgerows[graph->edge_rowstart(node, ii)];
      for ( unsigned j = 0; j < graph->nodesize[node]; j++ ) { // Over cols
        unsigned realj = rmn[j];
        for ( unsigned i = node == dest ? j : 0; i < maxj; i++ ) { // Over rows
          unsigned reali = rmd[edgerows[i]];
          unsigned idx = MI(i, j, 1, maxj);
          //fprintf(fh, "[%d %d] ", j, i);
          fprintf(fh, "%d %d %.*e\n", realj, reali,
                  DBL_DIG+3, matrix[idx]);
        }
      }
    }
    rmn += graph->nodesize[node];
  }
  if ( fclose(fh) != 0 ) abort();
}

struct PerThread
{
    double *tempmatrix;
    PerThread(unsigned int max_c_size) {
        tempmatrix = (double*) calloc(max_c_size, sizeof(double));
    }

    ~PerThread() {
        free((void*) tempmatrix);
    }
};

typedef galois::runtime::PerThreadStorage<PerThread> PTS;

template<typename Graph>
struct TreeExecModel
{
  typedef std::set<node_t> ChildList;
  typedef std::vector<ChildList> Children;
  Graph *graph;
  PTS *storage;
  Children children;
  ChildList rootnodes;
  int rootlock = 1;

  TreeExecModel (Graph *graph, PTS *storage): graph(graph), storage(storage), children(graph->nodecount, ChildList()) {
    node_t used[graph->nodecount];
    for (node_t i = 0; i < graph->nodecount; ++i) {
      used[i] = 0;
      if (graph->edge_begin(i) != graph->edge_end(i)-1) {
        for (edge_t ci = graph->edge_begin(i), ce=graph->edge_end(i);
             ci != ce; ++ci) {
          
          node_t dest = graph->edge_dest(i, ci);
          if (dest != i && used[i] == 0) {
            children[dest].insert(i);
            used[i]++;
          }
        }
      } else {
        // no outgoing edges? this must be a root!
        rootnodes.insert(i);
      }
    }
  }

  struct GaloisDivide {
    TreeExecModel *tm;
    
    GaloisDivide(TreeExecModel *tm) : tm(tm) { }

    template <typename C>
    void operator() (node_t i, C & ctx) {
    // small hack to prevent constructing too many forests. Only one node
    // is responsible for adding roots for all subtrees.
      if (__sync_val_compare_and_swap(&tm->rootlock, 1, 0)) {
        for (node_t j : tm->children[i]) {
          ctx.spawn(j);
        }
        for (node_t j : tm->rootnodes) {
          if (j != i) ctx.spawn(j);
        }
      } else {
        for (node_t j : tm->children[i]) {
         ctx.spawn(j);
        }
    }
    }
  };

  struct GaloisConquer {
    TreeExecModel *tm;
    GaloisConquer(TreeExecModel *tm) : tm(tm) {}
    
    void operator() (node_t i) {
      do_node(tm->graph, i, tm->storage->getLocal()->tempmatrix);
    }
  };

  void run() {
    galois::runtime::for_each_ordered_tree(*(rootnodes.begin()),
                    GaloisDivide(this),
                    GaloisConquer(this),
                    "SuperNodalCholesky");
  }  
};

int main(int argc, char *argv[]) {
  LonestarStart(argc, argv, 0,0,0); 
  galois::StatManager statManager;
  MutableGraph<true> *temp = new MutableGraph<true>;
  BidiGraph graph(*temp);
  delete temp;
  etree = make_etree(&graph);

  // Graph of supernodes
  SuperGraph sg;
  unsigned max_c_size;
  struct mapping *mapping;
  {
    struct timeval super_start, super_end;
    gettimeofday(&super_start, NULL);
    make_supergraph(&graph, &sg, &max_c_size, &mapping);
    gettimeofday(&super_end, NULL);
    printf("supernodes done in %ld ms\n", TIMEDELTA_MICRO(super_start, super_end)/1000);
  }

  // Do Cholesky
  //////////////////////////////////////////////////////////////////////


  galois::StatTimer T("NumericTime");
  PTS pts(max_c_size);
  TreeExecModel<SuperGraph> tm(&sg, &pts);
  papi_start();
  T.start();
  tm.run();
  T.stop();
  papi_stop("");
  {
    // Revmapping for printout (separated so it can be excluded from timing)
    unsigned *revmappingidx = (unsigned *)malloc(sg.nodecount*sizeof(unsigned));
    if ( !revmappingidx ) abort();
    unsigned *revmapping = (unsigned *)malloc(graph.nodecount*sizeof(unsigned));
    if ( !revmapping ) abort();
    // Fill indexing for revmapping
    revmappingidx[0] = sg.nodesize[0];
    for ( unsigned i = 1; i < sg.nodecount; i++ )
      revmappingidx[i] = revmappingidx[i-1] + sg.nodesize[i];
    // Fill revmapping
    for ( unsigned i = 0; i < graph.nodecount; i++ ) {
      struct mapping *m = &mapping[i];
      unsigned idx = m->n > 0 ? revmappingidx[m->n-1] : 0;
      revmapping[idx+m->p] = i;
    }
    print_supergraph(&sg, revmappingidx, revmapping);
  }

  return 0;
}
