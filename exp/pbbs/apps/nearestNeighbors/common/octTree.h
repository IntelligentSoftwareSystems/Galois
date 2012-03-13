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

#ifndef _BENCH_OCTTREE_INCLUDED
#define _BENCH_OCTTREE_INCLUDED

#include <iostream>
#include <cstdlib>
#include "sequence.h"
#include "parallel.h"
#include "geometry.h"
#include "blockRadixSort.h"
using namespace std;

// *************************************************************
//    QUAD/OCT TREE NODES
// *************************************************************

#define gMaxLeafSize 16  // number of points stored in each leaf

template <class vertex, class point>
struct nData {
  int cnt;
  nData(int x) : cnt(x) {}
  nData(point center) : cnt(0) {}
  nData& operator+=(const nData op) {cnt += op.cnt; return *this;}
  nData& operator+=(const vertex* op) {cnt += 1; return *this;}
};

template <class pointT, class vectT, class vertexT, class nodeData = nData<vertexT,pointT> >
class gTreeNode {
private :

public :
  typedef pointT point;
  typedef vectT fvect;
  typedef vertexT vertex;
  struct getPoint {point operator() (vertex* v) {return v->pt;}};
  struct minpt {point operator() (point a, point b) {return a.minCoords(b);}};
  struct maxpt {point operator() (point a, point b) {return a.maxCoords(b);}};

  point center; // center of the box
  double size;   // width of each dimension, which have to be the same
  nodeData data; // total mass of vertices in the box
  int count;  // number of vertices in the box
  gTreeNode *children[8];
  vertex **vertices;

  // wraps a bounding box around the points and generates
  // a tree.
  static gTreeNode* gTree(vertex** vv, int n) {

    // calculate bounding box
    point* pt = newA(point, n);
    // copying to an array of points to make reduce more efficient
//    {parallel_for(int i=0; i < n; i++) 
    {
      parallel_doall(int, i, 0, n)  {
	pt[i] = vv[i]->pt;
      } parallel_doall_end
    }

    point minPt = sequence::reduce(pt, n, minpt());
    point maxPt = sequence::reduce(pt, n, maxpt());
    free(pt);
    //cout << "min "; minPt.print(); cout << endl;
    //cout << "max "; maxPt.print(); cout << endl;
    fvect box = maxPt-minPt;
    point center = minPt+(box/2.0);

    // copy before calling recursive routine since recursive routine is destructive
    vertex** v = newA(vertex*,n);
//    {parallel_for(int i=0; i < n; i++) 
    {
      parallel_doall(int, i, 0, n)  {
	v[i] = vv[i];
      }
    } parallel_doall_end
    //cout << "about to build tree" << endl;

    gTreeNode* result = new gTreeNode(v, n, center, box.maxDim());
    free(v);
    return result;
  }

  int IsLeaf() { return (vertices != NULL); }

  void del() {
    if (IsLeaf()) delete [] vertices;
    else {
      for (int i=0 ; i < (1 << center.dimension()); i++) {
	children[i]->del();
	delete children[i];
      }
    }
  }

  // Returns the depth of the tree rooted at this node
  int Depth() {
    int depth;
    if (IsLeaf()) depth = 0;
    else {
      depth = 0;
      for (int i=0 ; i < (1 << center.dimension()); i++)
	depth = max(depth,children[i]->Depth());
    }
    return depth+1;
  }

  // Returns the size of the tree rooted at this node
  int Size() {
    int sz;
    if (IsLeaf()) {
      sz = count;
      for (int i=0; i < count; i++) 
	if (vertices[i] < ((vertex*) NULL)+1000) 
	  cout << "oops: " << vertices[i] << "," << count << "," << i << endl;
    }
    else {
      sz = 0;
      for (int i=0 ; i < (1 << center.dimension()); i++)
	sz += children[i]->Size();
    }
    return sz;
  }

  template <class F>
  void applyIndex(int s, F f) {
    if (IsLeaf())
      for (int i=0; i < count; i++) f(vertices[i],s+i);
    else {
      int ss = s;
      for (int i=0 ; i < (1 << center.dimension()); i++) {
	cilk_spawn children[i]->applyIndex(ss,f);
	ss += children[i]->count;
      }
      cilk_sync;
    }
  }

  struct flatten_FA {
    vertex** _A;
    flatten_FA(vertex** A) : _A(A) {}
    void operator() (vertex* p, int i) {
      _A[i] = p;}
  };

  vertex** flatten() {
    vertex** A = new vertex*[count];
    applyIndex(0,flatten_FA(A));
    return A;
  }

  // Returns the child the vertex p appears in
  int findQuadrant (vertex* p) {
    return (p->pt).quadrant(center); }

  // A hack to get around Cilk shortcomings
  static gTreeNode *newTree(vertex** S, int n, point cnt, double sz) {
    return new gTreeNode(S, n, cnt, sz); }

  // Used as a closure in collectD
  struct findChild {
    gTreeNode *tr;
    findChild(gTreeNode *t) : tr(t) {}
    int operator() (vertex* p) {
      int r = tr->findQuadrant(p);
      return r;}
  };

// the recursive routine for generating the tree -- actually mutually recursive
// due to newTree
  gTreeNode(vertex** S, int n, point cnt, double sz) : data(nodeData(cnt))
  {
    //cout << "n=" << n << endl;
    count = n;
    size = sz;
    center = cnt;
    vertices = NULL;
    int quadrants = (1 << center.dimension());

    if (count > gMaxLeafSize) {
      int offsets[8];
      intSort::iSort(S, offsets, n, quadrants, findChild(this));
      if (0) {
	for (int i=0; i < n; i++) {
	  cout << "  " << i << ":" << this->findQuadrant(S[i]);
	}
      }
      //for (int i=0; i < quadrants; i++)
      //cout << i << ":" << offsets[i] << "  ";
      //cout << endl;
      // Give each child its appropriate center and size
      // The centers are offset by size/4 in each of the dimensions
      for (int i=0 ; i < quadrants; i++) {
	point newcenter = center.offsetPoint(i, size/4.0);
	int l = ((i == quadrants-1) ? n : offsets[i+1]) - offsets[i];
	children[i] = cilk_spawn newTree(S + offsets[i], l, newcenter, size/2.0);
      }
      cilk_sync;
      data = nodeData(center);
      for (int i=0 ; i < quadrants; i++) 
	if (children[i]->count > 0)
	  data += children[i]->data;
    } else {
      vertices = new vertex*[count];
      data = nodeData(center);
      for (int i=0; i < count; i++) {
	vertex* p = S[i];
	data += p;
	vertices[i] = p;
      }
    }
  }
};

#endif // _BENCH_OCTTREE_INCLUDED
