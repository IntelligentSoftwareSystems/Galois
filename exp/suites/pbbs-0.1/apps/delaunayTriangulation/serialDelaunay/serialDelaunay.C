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

#include <vector>
#include "sequence.h"
#include "geometry.h"
#include "nearestNeighbors.h"
#include "delaunayDefs.h"
#include "topology.h"
using namespace std;

// *************************************************************
//    ROUTINES FOR FINDING AND INSERTING A NEW POINT
// *************************************************************

// Finds a vertex (p) in a mesh starting at any triangle (start)
// Requires that the mesh is properly connected and convex
simplex find(vertex *p, simplex start) {
  simplex t = start;
  while (1) {
    int i;
    for (i=0; i < 3; i++) {
      t = t.rotClockwise();
      if (t.outside(p)) {t = t.across(); break;}
    }
    if (i==3) return t;
  }
}

// Holds simplex queue used to store the cavity created 
// while searching from a vertex.
typedef vector<simplex> Qs;

// Recursive routine for finding a cavity across an edge with
// respect to a vertex p.
// The simplex has orientation facing the direction it is entered.
//
//         a
//         | \ --> recursive call
//   p --> |T c 
// enter   | / --> recursive call
//         b
//
//  If p is in circumcircle of T then 
//     add T to simplexQ, c to vertexQ, and recurse
void findCavity(simplex t, vertex *p, Qs* q) {
  if (t.inCirc(p)) {
    q->push_back(t);
    t = t.rotClockwise();
    findCavity(t.across(), p, q);
    t = t.rotClockwise();
    findCavity(t.across(), p, q);
  }
}

bool insert(vertex *v, simplex t, Qs *q) {
  for (int i=0; i < 3; i++) {
    findCavity(t.across(), v, q);
    t = t.rotClockwise();
  }

  tri* t1 = v->t;  // the memory for the two new triangles
  tri* t2 = t1 + 1;  

  // the following 3 lines do all the side effects to the mesh.
  t.split(v, t1, t2);
  for (int i = 0; i < q->size(); i++) {
    (*q)[i].flip();
  }

  q->clear();
  return 0;
}

// *************************************************************
//    CREATING BOUNDING SIMPLEX
// *************************************************************

simplex generateBoundary(point2d* P, int n, int bCount, vertex* v, tri* t) {
  point2d minP = P[0];
  point2d maxP = P[0];
  for (int i=1; i < n; i++) {
    minP = minP.minCoords(P[i]);
    maxP = maxP.maxCoords(P[i]);
  }

  double size = (maxP-minP).Length();
  double stretch = 10.0;
  double radius = stretch*size;
  point2d center = maxP + (maxP-minP)/2.0;

  point2d* boundaryP = newA(point2d, bCount);
  double pi = 3.14159;
  for (int i=0; i < bCount; i++) {
    double x = radius * cos(2*pi*((float) i)/((float) bCount));
    double y = radius * sin(2*pi*((float) i)/((float) bCount));
    boundaryP[i] = center + vect2d(x,y);
    v[i] = vertex(boundaryP[i], i + n);
  }

  simplex s = simplex(&v[0], &v[1], &v[2], t); 
  for (int i=3; i < bCount; i++)
    s = s.extend(&v[i], t+i-2);
  return s;
}

// *************************************************************
//    MAIN LOOP
// *************************************************************

void incrementallyAddPoints(vertex** v, int n, vertex* start) {

  Qs *qs = new Qs;

  // create a point location structure
  typedef kNearestNeighbor<vertex,1> KNN;
  KNN knn = KNN(&start, 1);
  int multiplier = 8; // when to regenerate
  int nextNN = multiplier;

  for (int j=n-1; j>=0; j--) {
    vertex *u = knn.nearest(v[j]);
    simplex t = find(v[j],simplex(u->t,0));
    insert(v[j],t,qs);
    if((n-j) >= nextNN && (n-j) < n/multiplier){
      knn.del();
      knn = KNN(v+j, (n-j));
      nextNN = nextNN*multiplier;
    }
  }
 
  knn.del();
  free(qs);
}

// *************************************************************
//    OUTPUT TRIAGLES
// *************************************************************

_seq<triangle> outputTriangles(tri* Triangs, int numTriangles) {
  // generate array of triangles for output
  triangle* rt = newA(triangle, numTriangles);
  int k = 0;
  for (int i=0; i < numTriangles; i++) {
    int id1 = Triangs[i].vtx[0]->id;
    int id2 = Triangs[i].vtx[1]->id;
    int id3 = Triangs[i].vtx[2]->id;
    if (id1 >= 0 && id2 >= 0 && id3 >= 0) 
      rt[k++] = triangle(id1,id2,id3);
  }

  return _seq<triangle>(rt,k);
}

// *************************************************************
//    DRIVER
// *************************************************************

triangles<point2d> delaunay(point2d* P, int n) {
  int boundarySize = 10;

  int numVertices = n + boundarySize;
  vertex** v = newA(vertex*, n); // don't need pointers to boundary
  vertex* vv = newA(vertex, numVertices);
  for (int i=0; i < n; i++) 
    v[i] = new (&vv[i]) vertex(P[i], i);

  // allocate all the triangles needed
  int numTriangles = 2 * n + (boundarySize - 2);
  tri* Triangs = newA(tri, numTriangles); 

  // give two triangles to each vertex
  for (int i=0; i < n; i++)
      v[i]->t = Triangs + 2*i; 

  // generate boundary simplex (use last triangle)
  simplex sBoundary = generateBoundary(P, n, boundarySize, vv + n, Triangs + 2*n);
  vertex* v0 = sBoundary.t->vtx[0];

  // main loop to add all points
  incrementallyAddPoints(v, n, v0);
  free(v);

  triangle* rt = newA(triangle, numTriangles);
  for (int i=0; i < numTriangles; i++) {
    vertex** vtx = Triangs[i].vtx;
    rt[i] = triangle(vtx[0]->id, vtx[1]->id, vtx[2]->id);
  }

  point2d* rp = newA(point2d, numVertices);
  for (int i=0; i < numVertices; i++) 
    rp[i] = vv[i].pt;
  free(Triangs);
  free(vv);

  return triangles<point2d>(numVertices, numTriangles, rp, rt);
}
