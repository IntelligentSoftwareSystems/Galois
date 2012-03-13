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

#include <algorithm>
#include "parallel.h"
#include "geometry.h"
#include "gettime.h"
#include "sequence.h"
#include "float.h"
#include "sampleSort.h"
#include "ray.h"
#include "kdTree.h"
#include "rayTriangleIntersect.h"
using namespace std;

int CHECK = 0;  // if set checks 10 rays against brute force method
int STATS = 0;  // if set prints out some tree statistics

// Constants for deciding when to stop recursion in building the KDTree
float CT = 6.0;
float CL = 1.25;
float maxExpand = 1.6;
int maxRecursionDepth = 25;

// Constant for switching to sequential versions
int minParallelSize = 500000;

typedef pointT::floatT floatT;
typedef _vect3d<floatT> vectT;
typedef triangles<pointT> trianglesT;
typedef ray<pointT> rayT;

float boxSurfaceArea(BoundingBox B) {
  float r0 = B[0].max-B[0].min;
  float r1 = B[1].max-B[1].min;
  float r2 = B[2].max-B[2].min;
  return 2*(r0*r1 + r1*r2 + r0*r2);
}

float epsilon = .0000001;
range fixRange(float minv, float maxv) {
  if (minv == maxv) return range(minv,minv+epsilon);
  else return range(minv,maxv);
}

inline float inBox(pointT p, BoundingBox B) {
  return (p.x >= (B[0].min - epsilon) && p.x <= (B[0].max + epsilon) &&
	  p.y >= (B[1].min - epsilon) && p.y <= (B[1].max + epsilon) &&
	  p.z >= (B[2].min - epsilon) && p.z <= (B[2].max + epsilon));
}

// sequential version of best cut
cutInfo bestCutSerial(event* E, range r, range r1, range r2, int n) {
  if (r.max - r.min == 0.0) return cutInfo(FLT_MAX, r.min, n, n);
  float area = 2 * (r1.max-r1.min) * (r2.max-r2.min);
  float diameter = 2 * ((r1.max-r1.min) + (r2.max-r2.min));

  // calculate cost of each possible split
  int inLeft = 0;
  int inRight = n/2;
  float minCost = FLT_MAX;
  int k = 0;
  int rn = inLeft;
  int ln = inRight;
  for (int i=0; i <n; i++) {
    float cost;
    if (IS_END(E[i])) inRight--;
    float leftLength = E[i].v - r.min;
    float leftArea = area + diameter * leftLength;
    float rightLength = r.max - E[i].v;
    float rightArea = area + diameter * rightLength;
    cost = (leftArea * inLeft + rightArea * inRight);
    if (cost < minCost) {
      rn = inRight;
      ln = inLeft;
      minCost = cost;
      k = i;
    }
    if (IS_START(E[i])) inLeft++;
  }
  return cutInfo(minCost, E[k].v, ln, rn);
}

// parallel version of best cut
cutInfo bestCut(event* E, range r, range r1, range r2, int n) {
  if (n < minParallelSize)
    return bestCutSerial(E, r, r1, r2, n);
  if (r.max - r.min == 0.0) return cutInfo(FLT_MAX, r.min, n, n);

  // area of two orthogonal faces
  float orthogArea = 2 * ((r1.max-r1.min) * (r2.max-r2.min));

  // length of diameter of orthogonal face
  float diameter = 2 * ((r1.max-r1.min) + (r2.max-r2.min));

  // count number that end before i
  int* upperC = newA(int,n);
//  parallel_for (int i=0; i <n; i++) upperC[i] = IS_END(E[i]);
  parallel_doall(int, i, 0, n) { upperC[i] = IS_END(E[i]); } parallel_doall_end
  int u = sequence::plusScan(upperC, upperC, n);

  // calculate cost of each possible split location
  float* cost = newA(float,n);
//  parallel_for (int i=0; i <n; i++) {
  parallel_doall(int, i, 0, n)  {
    int inLeft = i - upperC[i];
    int inRight = n/2 - (upperC[i] + IS_END(E[i]));
    float leftLength = E[i].v - r.min;
    float leftArea = orthogArea + diameter * leftLength;
    float rightLength = r.max - E[i].v;
    float rightArea = orthogArea + diameter * rightLength;
    cost[i] = (leftArea * inLeft + rightArea * inRight);
  } parallel_doall_end

  // find minimum across all (maxIndex with less is minimum)
  int k = sequence::maxIndex(cost,n,less<float>());

  float c = cost[k];
  int ln = k - upperC[k];
  int rn = n/2 - (upperC[k] + IS_END(E[k]));
  free(upperC); free(cost);
  return cutInfo(c, E[k].v, ln, rn);
}

typedef pair<_seq<event>, _seq<event> > eventsPair;

eventsPair splitEventsSerial(range* boxes, event* events, 
			     float cutOff, int n) {
  int l = 0;
  int r = 0;
  event* eventsLeft = newA(event,n);
  event* eventsRight = newA(event,n);
  for (int i=0; i < n; i++) {
    int b = GET_INDEX(events[i]);
    if (boxes[b].min < cutOff) {
      eventsLeft[l++] = events[i];
      if (boxes[b].max > cutOff) 
	eventsRight[r++] = events[i]; 
    } else eventsRight[r++] = events[i]; 
  }
  return eventsPair(_seq<event>(eventsLeft,l), 
		    _seq<event>(eventsRight,r));
}

eventsPair splitEvents(range* boxes, event* events, float cutOff, int n) {
  if (n < minParallelSize)
    return splitEventsSerial(boxes, events, cutOff, n);
  bool* lower = newA(bool,n);
  bool* upper = newA(bool,n);

//  parallel_for (int i=0; i <n; i++) {
  parallel_doall(int, i, 0, n)  {
    int b = GET_INDEX(events[i]);
    lower[i] = boxes[b].min < cutOff;
    upper[i] = boxes[b].max > cutOff;
  } parallel_doall_end

  _seq<event> L = sequence::pack(events, lower, n);
  _seq<event> R = sequence::pack(events, upper, n);
  free(lower); free(upper);

  return eventsPair(L,R);
}

// n is the number of events (i.e. twice the number of triangles)
treeNode* generateNode(Boxes boxes, Events events, BoundingBox B, 
		       int n, int maxDepth) {
  //cout << "n=" << n << " maxDepth=" << maxDepth << endl;
  if (n <= 2 || maxDepth == 0) 
    return new treeNode(events, n, B);

  // loop over dimensions and find the best cut across all of them
  cutInfo cuts[3];
  for (int d = 0; d < 3; d++) 
    cuts[d] = cilk_spawn bestCut(events[d], B[d], B[(d+1)%3], B[(d+2)%3], n);
  cilk_sync;

  int cutDim = 0;
  for (int d = 1; d < 3; d++) 
    if (cuts[d].cost < cuts[cutDim].cost) cutDim = d;

  range* cutDimRanges = boxes[cutDim];
  float cutOff = cuts[cutDim].cutOff;
  float area = boxSurfaceArea(B);
  float bestCost = CT + CL * cuts[cutDim].cost/area;
  float origCost = (float) (n/2);

  // quit recursion early if best cut is not very good
  if (bestCost >= origCost || 
      cuts[cutDim].numLeft + cuts[cutDim].numRight > maxExpand * n/2)
    return new treeNode(events, n, B);

  // declare structures for recursive calls
  BoundingBox BBL;
  for (int i=0; i < 3; i++) BBL[i] = B[i];
  BBL[cutDim] = range(BBL[cutDim].min, cutOff);
  event* leftEvents[3];
  int nl;

  BoundingBox BBR;
  for (int i=0; i < 3; i++) BBR[i] = B[i];
  BBR[cutDim] = range(cutOff, BBR[cutDim].max);
  event* rightEvents[3];
  int nr;

  // now split each event array to the two sides
  eventsPair X[3];
  for (int d = 0; d < 3; d++) 
     X[d] = cilk_spawn splitEvents(cutDimRanges, events[d], cutOff, n);
  cilk_sync;

  for (int d = 0; d < 3; d++) {
    leftEvents[d] = X[d].first.A;
    rightEvents[d] = X[d].second.A;
    if (d == 0) {
      nl = X[d].first.n;
      nr = X[d].second.n;
    } else if (X[d].first.n != nl || X[d].second.n != nr) {
      cout << "kdTree: mismatched lengths, something wrong" << endl;
      abort();
    }
  }

  // free old events and make recursive calls
  for (int i=0; i < 3; i++) free(events[i]);
  treeNode *L;
  treeNode *R;
  L = cilk_spawn generateNode(boxes, leftEvents, BBL, nl, maxDepth-1);
  R = generateNode(boxes, rightEvents, BBR, nr, maxDepth-1);
  cilk_sync;

  return new treeNode(L, R, cutDim, cutOff, B);
 }

int tcount = 0;
int ccount = 0;

// Given an a ray, a bounding box, and a sequence of triangles, returns the 
// index of the first triangle the ray intersects inside the box.
// The triangles are given by n indices I into the triangle array Tri.
// -1 is returned if there is no intersection
int findRay(rayT r, int* I, int n, triangles<pointT> Tri, BoundingBox B) {
  if (STATS) { tcount += n; ccount += 1;}
  pointT* P = Tri.P;
  floatT tMin = FLT_MAX;
  int k = -1;
  for (int i = 0; i < n; i++) {
    int j = I[i];
    triangle* tr = Tri.T + j;
    pointT m[3] = {P[tr->C[0]],  P[tr->C[1]],  P[tr->C[2]]};
    floatT t = rayTriangleIntersect(r, m);
    if (t > 0.0 && t < tMin && inBox(r.o + r.d*t, B)) {
      tMin = t;
      k = j;
    }
  }
  return k;
}

// Given a ray and a tree node find the index of the first triangle the 
// ray intersects inside the box represented by that node.
// -1 is returned if there is no intersection
int findRay(rayT r, treeNode* TN, trianglesT Tri) {
  //cout << "TN->n=" << TN->n << endl;
  if (TN->isLeaf()) 
    return findRay(r, TN->triangleIndices, TN->n, Tri, TN->box);
  pointT o = r.o;
  vectT d = r.d;

  floatT oo[3] = {o.x,o.y,o.z};
  floatT dd[3] = {d.x,d.y,d.z};

  // intersect ray with splitting plane
  int k0 = TN->cutDim;
  int k1 = (k0 == 2) ? 0 : k0+1;
  int k2 = (k0 == 0) ? 2 : k0-1;
  point2d o_p(oo[k1], oo[k2]);
  vect2d d_p(dd[k1], dd[k2]);
  // does not yet deal with dd[k0] == 0
  floatT scale = (TN->cutOff - oo[k0])/dd[k0];
  point2d p_i = o_p + d_p * scale;

  range rx = TN->box[k1];
  range ry = TN->box[k2];
  floatT d_0 = dd[k0];

  // decide which of the two child boxes the ray intersects
  enum {LEFT, RIGHT, BOTH};
  int recurseTo = LEFT;
  if      (p_i.x < rx.min) { if (d_p.x*d_0 > 0) recurseTo = RIGHT;}
  else if (p_i.x > rx.max) { if (d_p.x*d_0 < 0) recurseTo = RIGHT;}
  else if (p_i.y < ry.min) { if (d_p.y*d_0 > 0) recurseTo = RIGHT;}
  else if (p_i.y > ry.max) { if (d_p.y*d_0 < 0) recurseTo = RIGHT;}
  else recurseTo = BOTH;

  if (recurseTo == RIGHT) return findRay(r, TN->right, Tri);
  else if (recurseTo == LEFT) return findRay(r, TN->left, Tri);
  else if (d_0 > 0) {
    int t = findRay(r, TN->left, Tri);
    if (t >= 0) return t;
    else return findRay(r, TN->right, Tri);
  } else {
    int t = findRay(r, TN->right, Tri);
    if (t >= 0) return t;
    else return findRay(r, TN->left, Tri);
  }
}

void processRays(trianglesT Tri, rayT* rays, int numRays, 
		 treeNode* R, int* results) {
//  parallel_for (int i= 0; i < numRays; i++) 
  parallel_doall(int, i, 0, numRays) { 
    results[i] = findRay(rays[i], R, Tri);
  } parallel_doall_end
}

int* rayCast(triangles<pointT> Tri, ray<pointT>* rays, int numRays) {
  startTime();

  // Extract triangles into a separate array for each dimension with
  // the lower and upper bound for each triangle in that dimension.
  Boxes boxes;
  int n = Tri.numTriangles;
  for (int d = 0; d < 3; d++) boxes[d] = newA(range, n);
  pointT* P = Tri.P;
//  parallel_for (int i=0; i < n; i++) {
  parallel_doall(int, i, 0, n)  {
    pointT p0 = P[Tri.T[i].C[0]];
    pointT p1 = P[Tri.T[i].C[1]];
    pointT p2 = P[Tri.T[i].C[2]];
    boxes[0][i] = fixRange(min(p0.x,min(p1.x,p2.x)),max(p0.x,max(p1.x,p2.x)));
    boxes[1][i] = fixRange(min(p0.y,min(p1.y,p2.y)),max(p0.y,max(p1.y,p2.y)));
    boxes[2][i] = fixRange(min(p0.z,min(p1.z,p2.z)),max(p0.z,max(p1.z,p2.z)));
  } parallel_doall_end

  // Loop over the dimensions creating an array of events for each
  // dimension, sorting each one, and extracting the bounding box
  // from the first and last elements in the sorted events in each dim.
  Events events;
  BoundingBox boundingBox;
  for (int d = 0; d < 3; d++) {
    events[d] = newA(event, 2*n); // freed while generating tree
//    parallel_for (int i=0; i <n; i++) {
    parallel_doall(int, i, 0, n)  {
      events[d][2*i] = event(boxes[d][i].min, i, START);
      events[d][2*i+1] = event(boxes[d][i].max, i, END);
    } parallel_doall_end
    compSort(events[d], n*2, cmpVal());
    boundingBox[d] = range(events[d][0].v, events[d][2*n-1].v);
  }
  nextTime("generate and sort events");

  // build the tree
  int recursionDepth = min(maxRecursionDepth, utils::log2Up(n)-1);
  treeNode* R = generateNode(boxes, events, boundingBox, n*2, 
			    recursionDepth);
  nextTime("build tree");

  if (STATS)
    cout << "Triangles across all leaves = " << R->n 
	 << " Leaves = " << R->leaves << endl;
  for (int d = 0; d < 3; d++) free(boxes[d]);

  // get the intersections
  int* results = newA(int,numRays);
  processRays(Tri, rays, numRays, R, results);
  nextTime("intersect rays");
  treeNode::del(R);
  nextTime("delete tree");

  if (CHECK) {
    int nr = 10;
    int* indx = newA(int,n);
//    parallel_for (int i= 0; i < n; i++) indx[i] = i;
    parallel_doall(int, i, 0, n) { indx[i] = i; } parallel_doall_end
    for (int i= 0; i < nr; i++) {
      cout << results[i] << endl;
      if (findRay(rays[i], indx, n, Tri, boundingBox) != results[i]) {
	cout << "bad intersect in checking ray intersection" << endl;
	abort();
      }
    }
  }

  if (STATS)
    cout << "tcount=" << tcount << " ccount=" << ccount << endl;
  return results;
}
