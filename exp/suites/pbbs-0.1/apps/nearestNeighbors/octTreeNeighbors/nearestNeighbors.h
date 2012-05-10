// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <limits>
#include "sequence.h"
#include "parallel.h"
#include "octTree.h"
//#include "ppOctTree.h"
using namespace std;

// A k-nearest neighbor structure
// requires vertexT to have pointT and vectT typedefs
template <class vertexT, int maxK>
struct kNearestNeighbor {
  typedef vertexT vertex;
  typedef typename vertexT::pointT point;
  typedef typename point::vectT fvect;

  typedef gTreeNode<point,fvect,vertex> qoTree;
  qoTree *tree;

  // generates the search structure
  kNearestNeighbor(vertex** vertices, int n) {
    tree = qoTree::gTree(vertices, n);
  }

  // returns the vertices in the search structure, in an
  //  order that has spacial locality
  vertex** vertices() {
    return tree->flatten();
  }

  void del() {tree->del();}

  struct kNN {
    vertex *ps;  // the element for which we are trying to find a NN
    vertex *pn[maxK];  // the current k nearest neighbors (nearest last)
    double rn[maxK]; // radius of current k nearest neighbors
    int quads;
    int k;
    kNN() {}

    // returns the ith smallest element (0 is smallest) up to k-1
    vertex* operator[] (const int i) { return pn[k-i-1]; }

    kNN(vertex *p, int kk) {
      if (kk > maxK) {cout << "k too large in kNN" << endl; abort();}
      k = kk;
      quads = (1 << (p->pt).dimension());
      ps = p;
      for (int i=0; i<k; i++) {
	pn[i] = (vertex*) NULL; 
	rn[i] = numeric_limits<double>::max();
      }
    }

    // if p is closer than pn then swap it in
    void update(vertex *p) { 
      //inter++;
      point opt = (p->pt);
      fvect v = (ps->pt) - opt;
      double r = v.Length();
      if (r < rn[0]) {
	pn[0]=p; rn[0] = r;
	for (int i=1; i < k && rn[i-1]<rn[i]; i++) {
	  swap(rn[i-1],rn[i]); swap(pn[i-1],pn[i]); }
      }
    }

    // looks for nearest neighbors in boxes for which ps is not in
    void nearestNghTrim(qoTree *T) {
      if (!(T->center).outOfBox(ps->pt, (T->size/2)+rn[0]))
	if (T->IsLeaf())
	  for (int i = 0; i < T->count; i++) update(T->vertices[i]);
	else 
	  for (int j=0; j < quads; j++) nearestNghTrim(T->children[j]);
    }

    // looks for nearest neighbors in box for which ps is in
    void nearestNgh(qoTree *T) {
      if (T->IsLeaf())
	for (int i = 0; i < T->count; i++) {
	  vertex *pb = T->vertices[i];
	  if (pb != ps) update(pb);
	}
      else {
	int i = T->findQuadrant(ps);
	nearestNgh(T->children[i]);
	for (int j=0; j < quads; j++) 
	  if (j != i) nearestNghTrim(T->children[j]);
      }
    }
  };

  vertex* nearest(vertex *p) {
    kNN nn(p,1);
    nn.nearestNgh(tree); 
    return nn[0];
  }

  // version that writes into result
  void kNearest(vertex *p, vertex** result, int k) {
    kNN nn(p,k);
    nn.nearestNgh(tree); 
    for (int i=0; i < k; i++) result[i] = 0;
    for (int i=0; i < k; i++) result[i] = nn[i];
  }

  // version that allocates result
  vertex** kNearest(vertex *p, int k) {
    vertex** result = newA(vertex*,k);
    kNearest(p,result,k);
    return result;
  }

};

// find the k nearest neighbors for all points in tree
// places pointers to them in the .ngh field of each vertex
template <int maxK, class vertexT>
void ANN(vertexT** v, int n, int k) {
  typedef kNearestNeighbor<vertexT,maxK> kNNT;

  kNNT T = kNNT(v, n);

  //cout << "built tree" << endl;

  // this reorders the vertices for locality
  vertexT** vr = T.vertices();

  // find nearest k neighbors for each point
  parallel_doall_1 (int, i, 0, n) {
    T.kNearest(vr[i], vr[i]->ngh, k);
  } parallel_doall_end

  free(vr);
  T.del();
}
