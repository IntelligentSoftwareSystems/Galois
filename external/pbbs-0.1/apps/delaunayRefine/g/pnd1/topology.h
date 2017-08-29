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

#ifndef _TOPOLOGY_INCLUDED
#define _TOPOLOGY_INCLUDED

#include <iostream>
#include "geometry.h"

#include <pthread.h>

#ifdef GALOIS_USE_DMP
#include "dmp.h"
#endif

using namespace std;

// *************************************************************
//    TOPOLOGY
// *************************************************************

struct vertex;

// an unoriented triangle with its three neighbors and 3 vertices
//          vtx[1]
//           o 
//           | \ -> ngh[1]
// ngh[2] <- |   o vtx[0]
//           | / -> ngh[0]
//           o
//         vtx[2]
struct tri {
  tri *ngh [3];
  vertex *vtx [3];
  int id;
  bool initialized;
  char bad;  // used to mark badly shaped triangles
  void setT(tri *t1, tri *t2, tri* t3) {
    ngh[0] = t1; ngh[1] = t2; ngh[2] = t3; }
  void setV(vertex *v1, vertex *v2, vertex *v3) {
    vtx[0] = v1; vtx[1] = v2; vtx[2] = v3; }
  int locate(tri *t) {
    for (int i=0; i < 3; i++)
      if (ngh[i] == t) return i;
    cout<<"did not locate back pointer in triangulation\n";
    abort(); // did not find
  }
  void update(tri *t, tri *tn) {
    for (int i=0; i < 3; i++)
      if (ngh[i] == t) {ngh[i] = tn; return;}
    cout<<"did not update\n";
    abort(); // did not find
  }
};

// a vertex pointing to an arbitrary triangle to which it belongs (if any)
struct vertex {
  typedef point2d pointT;
  point2d pt;
  pthread_mutex_t lock;
  tri *t;
  tri *badT;
  int id;
  int reserve;
  void print() {
    cout << id << " (" << pt.x << "," << pt.y << ") " << endl;
  }
  vertex(point2d p, int i) : pt(p), id(i), reserve(-1)
			   , badT(NULL)
  {  
    pthread_mutex_init(&lock, NULL);
  }
  ~vertex() {
    pthread_mutex_destroy(&lock);
  }
  int acquire(int mark) {
    int ret = 0;
    pthread_mutex_lock(&lock);
    if (reserve == -1) {
      reserve = mark;
      ret = 1;
    } else if (reserve == mark) {
      ret = 2;
    }
    pthread_mutex_unlock(&lock);
    return ret;
  }
};

inline int mod3(int i) {return (i>2) ? i-3 : i;}

// a simplex is just an oriented triangle.  An integer (o)
// is used to indicate which of 3 orientations it is in (0,1,2)
// If boundary is set then it represents the edge through t.ngh[o],
// which is a NULL pointer.
struct simplex {
  tri *t;
  int o;
  bool boundary;
  bool failed;
  simplex(tri *tt, int oo) : t(tt), o(oo), boundary(0), failed(false) {}
  simplex(tri *tt, int oo, bool _b) : t(tt), o(oo), boundary(_b), failed(false) {}
  simplex(vertex *v1, vertex *v2, vertex *v3, tri *tt): failed(false) {
    t = tt;
    t->ngh[0] = t->ngh[1] = t->ngh[2] = NULL;
    t->vtx[0] = v1; v1->t = t;
    t->vtx[1] = v2; v2->t = t;
    t->vtx[2] = v3; v3->t = t;
    o = 0;
    boundary = 0;
  }

  void print() {
    if (t == NULL) cout << "NULL simp" << endl;
    else {
      cout << "vtxs=";
      for (int i=0; i < 3; i++) 
	if (t->vtx[mod3(i+o)] != NULL)
	  cout << t->vtx[mod3(i+o)]->id << " (" <<
	    t->vtx[mod3(i+o)]->pt.x << "," <<
	    t->vtx[mod3(i+o)]->pt.y << ") ";
	else cout << "NULL ";
      cout << endl;
    }
  }

  template<typename QsTy>
  bool acquire(vertex* v, int id, QsTy* q) {
    int ret = v->acquire(id);
    if (ret == 1 && q != NULL)
      q->acquireQ.push_back(v);
    return ret != 0;
  }

  template<typename QsTy>
  bool acquire(int id, QsTy* q) {
    for (int i = 0; i < 3; ++i) {
      if (!acquire(t->vtx[i], id, q))
        return false;
    }
    return true;
  }

  template<typename QsTy>
  simplex across(int id, QsTy* q) {
    if (!acquire(t->vtx[o], id, q)) {
      failed = true;
      return simplex(t, o, 1);
    }

    tri *to = t->ngh[o];
    
    if (to != NULL) {
      for (int i = 0; i < 3; ++i) {
        if (!acquire(to->vtx[i], id, q)) {
          failed = true;
          return simplex(t, o, 1);
        }
      }
      return simplex(to, to->locate(t));
    }
    else return simplex(t,o,1);
  }

  simplex across() {
    tri *to = t->ngh[o];
    if (to != NULL) return simplex(to,to->locate(t));
    else return simplex(t,o,1);
  }

  // depending on initial triangle this could be counterclockwise
  simplex rotClockwise() { return simplex(t,mod3(o+1));}

  bool valid() {return (!boundary);}
  bool isTriangle() {return (!boundary);}
  bool isBoundary() {return boundary;}
  
  vertex *firstVertex() {return t->vtx[o];}

  bool inCirc(vertex *v) {
    if (boundary || t == NULL) return 0;
    return inCircle(t->vtx[0]->pt, t->vtx[1]->pt, 
		    t->vtx[2]->pt, v->pt);
  }

  // the angle facing the across edge
  double farAngle() {
    return angle(t->vtx[mod3(o+1)]->pt,
		 t->vtx[o]->pt,
		 t->vtx[mod3(o+2)]->pt);
  }

  bool outside(vertex *v) {
    if (boundary || t == NULL) return 0;
    return counterClockwise(t->vtx[mod3(o+2)]->pt, v->pt, t->vtx[o]->pt);
  }

  // flips two triangles and adjusts neighboring triangles
  void flip() { 
    simplex s = across();
    int o1 = mod3(o+1);
    int os1 = mod3(s.o+1);

    tri *t1 = t->ngh[o1];
    tri *t2 = s.t->ngh[os1];
    vertex *v1 = t->vtx[o1];
    vertex *v2 = s.t->vtx[os1];

    t->vtx[o]->t = s.t;
    t->vtx[o] = v2;
    t->ngh[o] = t2;
    if (t2 != NULL) t2->update(s.t,t);
    t->ngh[o1] = s.t;

    s.t->vtx[s.o]->t = t;
    s.t->vtx[s.o] = v1;
    s.t->ngh[s.o] = t1;
    if (t1 != NULL) t1->update(t,s.t);
    s.t->ngh[os1] = t;
  }

  // splits the triangle into three triangles with new vertex v in the middle
  // updates all neighboring simplices
  // ta0 and ta0 are pointers to the memory to use for the two new triangles
  void split(vertex* v, tri* ta0, tri* ta1) {
    v->t = t;
    tri *t1 = t->ngh[0]; tri *t2 = t->ngh[1]; tri *t3 = t->ngh[2];
    vertex *v1 = t->vtx[0]; vertex *v2 = t->vtx[1]; vertex *v3 = t->vtx[2];
    t->ngh[1] = ta0;        t->ngh[2] = ta1;
    t->vtx[1] = v;
    ta0->setT(t2,ta1,t);  ta0->setV(v2,v,v1);
    ta1->setT(t3,t,ta0);  ta1->setV(v3,v,v2);
    if (t2 != NULL) t2->update(t,ta0);      
    if (t3 != NULL) t3->update(t,ta1);
    v2->t = ta0;
  }

  // splits one of the boundaries of a triangle to form two triangles
  // the orientation dictates which edge to split (i.e., t.ngh[o])
  // ta is a pointer to memory to use for the new triangle
  void splitBoundary(vertex* v, tri* ta) {
    int o1 = mod3(o+1);
    int o2 = mod3(o+2);
    if (t->ngh[o] != NULL) {
      cout << "simplex::splitBoundary: not boundary" << endl; abort();}
    v->t = t;
    tri *t2 = t->ngh[o2];
    vertex *v1 = t->vtx[o1]; vertex *v2 = t->vtx[o2];
    t->ngh[o2] = ta;   t->vtx[o2] = v;
    ta->setT(t2,NULL,t);  ta->setV(v2,v,v1);
    if (t2 != NULL) t2->update(t,ta);      
    v2->t = t;
  }

  // given a vertex v, extends an boundary edge (t.ngh[o]) with an extra 
  // triangle on that edge with apex v.  
  // ta is used as the memory for the triangle
  simplex extend(vertex* v, tri* ta) {
    if (t->ngh[o] != NULL) {
      cout << "simplex::extend: not boundary" << endl; abort();}
    t->ngh[o] = ta;
    ta->setV(t->vtx[o], t->vtx[mod3(o+2)], v);
    ta->setT(NULL,t,NULL);
    v->t = ta;
    return simplex(ta,0);
  }

};

// this might or might not be needed
void topologyFromTriangles(triangles<point2d> Tri, vertex** vr, tri** tr);

#endif // _TOPOLOGY_INCLUDED
