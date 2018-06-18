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
#include "gettime.h"
#include "parallel.h"
#include "delaunay.h"
#include "delaunayDefs.h"
#include "topology.h"

using namespace std;

// if on verifies the Delaunay is correct
#define CHECK 0

// *************************************************************
//    ROUTINES FOR FINDING AND INSERTING A NEW POINT
// *************************************************************

// Finds a vertex (p) in a mesh starting at any triangle (start)
// Requires that the mesh is properly connected and convex
simplex find(vertex* p, simplex start) {
  simplex t = start;
  while (1) {
    int i;
    for (i = 0; i < 3; i++) {
      t = t.rotClockwise();
      if (t.outside(p)) {
        t = t.across();
        break;
      }
    }
    if (i == 3)
      return t;
    if (!t.valid())
      return t;
  }
}

// Holds vertex and simplex queues used to store the cavity created
// while searching from a vertex between when it is initially searched
// and later checked to see if all corners are reserved.
struct Qs {
  vector<vertex*> vertexQ;
  vector<simplex> simplexQ;
};

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
void findCavity(simplex t, vertex* p, Qs* q) {
  if (t.inCirc(p)) {
    q->simplexQ.push_back(t);
    t = t.rotClockwise();
    findCavity(t.across(), p, q);
    q->vertexQ.push_back(t.firstVertex());
    t = t.rotClockwise();
    findCavity(t.across(), p, q);
  }
}

// Finds the cavity for v and tries to reserve vertices on the
// boundary (v must be inside of the simplex t)
// The boundary vertices are pushed onto q->vertexQ and
// simplices to be deleted on q->simplexQ (both initially empty)
// It makes no side effects to the mesh other than to X->reserve
void reserveForInsert(vertex* v, simplex t, Qs* q) {
  // each iteration searches out from one edge of the triangle
  for (int i = 0; i < 3; i++) {
    q->vertexQ.push_back(t.firstVertex());
    findCavity(t.across(), v, q);
    t = t.rotClockwise();
  }
  // the maximum id new vertex that tries to reserve a boundary vertex
  // will have its id written.  reserve starts out as -1
  for (int i = 0; i < q->vertexQ.size(); i++)
    utils::writeMax(&((q->vertexQ)[i]->reserve), v->id);
}

// checks if v "won" on all adjacent vertices and inserts point if so
bool insert(vertex* v, simplex t, Qs* q) {
  bool flag = 0;
  for (int i = 0; i < q->vertexQ.size(); i++) {
    vertex* u = (q->vertexQ)[i];
    if (u->reserve == v->id)
      u->reserve = -1; // reset to -1
    else
      flag = 1; // someone else with higher priority reserved u
  }
  if (!flag) {
    tri* t1 = v->t; // the memory for the two new triangles
    tri* t2 = t1 + 1;
    // the following 3 lines do all the side effects to the mesh.
    t.split(v, t1, t2);
    for (int i = 0; i < q->simplexQ.size(); i++) {
      (q->simplexQ)[i].flip();
    }
  }
  q->simplexQ.clear();
  q->vertexQ.clear();
  return flag;
}

// *************************************************************
//    CHECKING THE TRIANGULATION
// *************************************************************

void checkDelaunay(tri* triangs, int n, int boundarySize) {
  int* bcount = newA(int, n);
  //  parallel_for(int j=0; j<n; j++) bcount[j] = 0;
  parallel_doall(int, j, 0, n) { bcount[j] = 0; }
  parallel_doall_end
  //  parallel_for (int i=0; i<n; i++) {
  parallel_doall(int, i, 0, n) {
    if (triangs[i].initialized >= 0) {
      simplex t = simplex(&triangs[i], 0);
      for (int i = 0; i < 3; i++) {
        simplex a = t.across();
        if (a.valid()) {
          vertex* v = a.rotClockwise().firstVertex();
          if (!t.outside(v)) {
            cout << "Inside Out: ";
            v->pt.print();
            t.print();
          }
          if (t.inCirc(v)) {
            cout << "In Circle Violation: ";
            v->pt.print();
            t.print();
          }
        } else
          bcount[i]++;
        t = t.rotClockwise();
      }
    }
  }
  parallel_doall_end if (boundarySize != sequence::plusReduce(bcount, n)) cout
      << "Wrong boundary size: should be " << boundarySize << " is " << bcount
      << endl;
  free(bcount);
}

// *************************************************************
//    CREATING BOUNDING SIMPLEX
// *************************************************************

struct minpt {
  point2d operator()(point2d a, point2d b) { return a.minCoords(b); }
};
struct maxpt {
  point2d operator()(point2d a, point2d b) { return a.maxCoords(b); }
};

simplex generateBoundary(point2d* P, int n, int bCount, vertex* v, tri* t) {
  point2d minP   = sequence::reduce(P, n, minpt());
  point2d maxP   = sequence::reduce(P, n, maxpt());
  double size    = (maxP - minP).Length();
  double stretch = 10.0;
  double radius  = stretch * size;
  point2d center = maxP + (maxP - minP) / 2.0;

  point2d* boundaryP = newA(point2d, bCount);
  double pi          = 3.14159;
  for (int i = 0; i < bCount; i++) {
    double x     = radius * cos(2 * pi * ((float)i) / ((float)bCount));
    double y     = radius * sin(2 * pi * ((float)i) / ((float)bCount));
    boundaryP[i] = center + vect2d(x, y);
    v[i]         = vertex(boundaryP[i], i + n);
  }

  simplex s = simplex(&v[0], &v[1], &v[2], t);
  for (int i = 3; i < bCount; i++)
    s = s.extend(&v[i], t + i - 2);
  return s;
}

// *************************************************************
//    MAIN LOOP
// *************************************************************

void incrementallyAddPoints(vertex** v, int n, vertex* start) {
  int numRounds = Exp::getNumRounds();
  numRounds     = numRounds <= 0 ? 100 : numRounds;

  // various structures needed for each parallel insertion
  // int maxR = (int) (n/100) + 1; // maximum number to try in parallel
  int maxR = (int)(n / numRounds) + 1; // maximum number to try in parallel
  Qs* qqs  = newA(Qs, maxR);
  Qs** qs  = newA(Qs*, maxR);
  for (int i = 0; i < maxR; i++) {
    qs[i] = new (&qqs[i]) Qs;
  }
  simplex* t  = newA(simplex, maxR);
  bool* flags = newA(bool, maxR);
  vertex** h  = newA(vertex*, maxR);

  // create a point location structure
  typedef kNearestNeighbor<vertex, 1> KNN;
  KNN knn        = KNN(&start, 1);
  int multiplier = 8; // when to regenerate
  int nextNN     = multiplier;

  int top    = n;
  int rounds = 0;
  int failed = 0;

  // process all vertices starting just below top
  while (top > 0) {

    // every once in a while create a new point location
    // structure using all points inserted so far
    if ((n - top) >= nextNN && (n - top) < n / multiplier) {
      knn.del();
      knn    = KNN(v + top, n - top);
      nextNN = nextNN * multiplier;
    }

    // determine how many vertices to try in parallel
    // int cnt = 1 + (n-top)/100;  // 100 is pulled out of a hat
    int cnt     = 1 + (n - top) / numRounds;
    cnt         = (cnt > maxR) ? maxR : cnt;
    cnt         = (cnt > top) ? top : cnt;
    vertex** vv = v + top - cnt;

    // for trial vertices find containing triangle, determine cavity
    // and reserve vertices on boundary of cavity
    //    parallel_for (int j = 0; j < cnt; j++) {
    parallel_doall(int, j, 0, cnt) {
      vertex* u = knn.nearest(vv[j]);
      t[j]      = find(vv[j], simplex(u->t, 0));
      reserveForInsert(vv[j], t[j], qs[j]);
    }
    parallel_doall_end

    // For trial vertices check if they own their boundary and
    // update mesh if so.  flags[i] is 1 if failed (need to retry)
    //    parallel_for (int j = 0; j < cnt; j++) {
    parallel_doall(int, j, 0, cnt) {
      flags[j] = insert(vv[j], t[j], qs[j]);
    }
    parallel_doall_end

        // Pack failed vertices back onto Q and successful
        // ones up above (needed for point location structure)
        int k = sequence::pack(vv, h, flags, cnt);
    //    parallel_for (int j = 0; j < cnt; j++) flags[j] = !flags[j];
    parallel_doall(int, j, 0, cnt) { flags[j] = !flags[j]; }
    parallel_doall_end sequence::pack(vv, h + k, flags, cnt);
    //    parallel_for (int j = 0; j < cnt; j++) vv[j] = h[j];
    parallel_doall(int, j, 0, cnt) { vv[j] = h[j]; }
    parallel_doall_end

        failed += k;
    top = top - cnt + k; // adjust top, accounting for failed vertices
    rounds++;
  }

  knn.del();
  free(qqs);
  free(qs);
  free(t);
  free(flags);
  free(h);

  cout << "n=" << n << "  Total retries=" << failed
       << "  Total rounds=" << rounds << endl;
}

// *************************************************************
//    DRIVER
// *************************************************************

triangles<point2d> delaunay(point2d* P, int n) {
  int boundarySize = 10;
  startTime();
  // allocate space for vertices
  int numVertices = n + boundarySize;
  vertex** v      = newA(vertex*, n); // don't need pointers to boundary
  vertex* vv      = newA(vertex, numVertices);
  //  {parallel_for (int i=0; i < n; i++)
  {
    parallel_doall(int, i, 0, n) { v[i] = new (&vv[i]) vertex(P[i], i); }
    parallel_doall_end
  }

  // allocate all the triangles needed
  int numTriangles = 2 * n + (boundarySize - 2);
  tri* Triangs     = newA(tri, numTriangles);

  // give two triangles to each vertex
  //  {parallel_for (int i=0; i < n; i++)
  {parallel_doall(int, i, 0, n){v[i]->t = Triangs + 2 * i;
}
parallel_doall_end
}

// generate boundary simplex (use last triangle)
simplex sBoundary =
    generateBoundary(P, n, boundarySize, vv + n, Triangs + 2 * n);
vertex* v0 = sBoundary.t->vtx[0];

nextTime("initialize");
// main loop to add all points
incrementallyAddPoints(v, n, v0);
free(v);
nextTime("add points");

if (CheckResult) {
  checkDelaunay(Triangs, numTriangles, boundarySize);
  cout << "result ok\n";
}

triangle* rt = newA(triangle, numTriangles);
//  parallel_for (int i=0; i < numTriangles; i++) {
parallel_doall(int, i, 0, numTriangles) {
  vertex** vtx = Triangs[i].vtx;
  rt[i]        = triangle(vtx[0]->id, vtx[1]->id, vtx[2]->id);
}
parallel_doall_end

    point2d* rp = newA(point2d, numVertices);
//  parallel_for (int i=0; i < numVertices; i++)
parallel_doall(int, i, 0, numVertices) { rp[i] = vv[i].pt; }
parallel_doall_end free(Triangs);
free(vv);
nextTime("generate output");

return triangles<point2d>(numVertices, numTriangles, rp, rt);
}
