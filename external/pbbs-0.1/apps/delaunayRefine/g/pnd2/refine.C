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
#include <vector>
#include <deque>
#include "sequence.h"
#include "gettime.h"
#include "parallel.h"
#include "refine.h"
#include "topology.h"

using namespace std;

// *************************************************************
//   PARALLEL HASH TABLE TO STORE WORK QUEUE OF SKINNY TRIANGLES
// *************************************************************

#include "deterministicHash.h"

struct hashTriangles {
  typedef tri* eType;
  typedef tri* kType;
  eType empty() { return NULL; }
  kType getKey(eType v) { return v; }
  unsigned int hash(kType s) { return utils::hash(s->id); }
  int cmp(kType s, kType s2) {
    return (s->id > s2->id) ? 1 : ((s->id == s2->id) ? 0 : -1);
  }
  bool replaceQ(eType s, eType s2) { return 0; }
};

typedef Table<hashTriangles> TriangleTable;
TriangleTable makeTriangleTable(int m) {
  return TriangleTable(m, hashTriangles());
}

// *************************************************************
//   THESE ARE TAKEN FROM delaunay.C
//   Perhaps should be #included
// *************************************************************

// Holds vertex and simplex queues used to store the cavity created
// while searching from a vertex between when it is initially searched
// and later checked to see if all corners are reserved.
struct Qs {
  vector<vertex*> vertexQ;
  vector<simplex> simplexQ;
  vector<vertex*> acquireQ;
  deque<int> abortedQ;
  vector<int> newQ;
  int aborted;
  int numBad;
  int begin;
  int end;
  Qs() : aborted(0), numBad(0) {}
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
bool findCavity(simplex t, vertex* p, int id, Qs* q) {
  if (!t.acquire(id, q))
    return false;

  if (t.inCirc(p)) {
    q->simplexQ.push_back(t);
    t = t.rotClockwise();

    simplex tt = t.across(id, q);
    if (t.failed)
      return false;

    if (!findCavity(tt, p, id, q))
      return false;

    q->vertexQ.push_back(t.firstVertex());
    t = t.rotClockwise();

    tt = t.across(id, q);
    if (t.failed)
      return false;

    if (!findCavity(tt, p, id, q))
      return false;
  }
  return true;
}

// Finds the cavity for v and tries to reserve vertices on the
// boundary (v must be inside of the simplex t)
// The boundary vertices are pushed onto q->vertexQ and
// simplices to be deleted on q->simplexQ (both initially empty)
// It makes no side effects to the mesh other than to X->reserve
bool reserveForInsert(vertex* v, simplex t, int id, Qs* q) {
  // each iteration searches out from one edge of the triangle
  for (int i = 0; i < 3; i++) {
    q->vertexQ.push_back(t.firstVertex());

    simplex tt = t.across(id, q);
    if (t.failed)
      return false;
    if (!findCavity(tt, v, id, q))
      return false;

    t = t.rotClockwise();
  }
  for (int i = 0; i < q->vertexQ.size(); i++) {
    if (!t.acquire(q->vertexQ[i], v->id, q))
      return false;
  }

  return true;
}

// *************************************************************
//   DEALING WITH THE CAVITY
// *************************************************************

inline bool skinnyTriangle(tri* t) {
  double minAngle = 30;
  if (minAngleCheck(t->vtx[0]->pt, t->vtx[1]->pt, t->vtx[2]->pt, minAngle))
    return 1;
  return 0;
}

inline bool obtuse(simplex t) {
  int o      = t.o;
  point2d p0 = t.t->vtx[(o + 1) % 3]->pt;
  vect2d v1  = t.t->vtx[o]->pt - p0;
  vect2d v2  = t.t->vtx[(o + 2) % 3]->pt - p0;
  return (v1.dot(v2) < 0.0);
}

inline point2d circumcenter(simplex t) {
  if (t.isTriangle())
    return triangleCircumcenter(t.t->vtx[0]->pt, t.t->vtx[1]->pt,
                                t.t->vtx[2]->pt);
  else { // t.isBoundary()
    point2d p0 = t.t->vtx[(t.o + 2) % 3]->pt;
    point2d p1 = t.t->vtx[t.o]->pt;
    return p0 + (p1 - p0) / 2.0;
  }
}

// this side affects the simplex by moving it into the right orientation
// and setting the boundary if the circumcenter encroaches on a boundary
inline bool checkEncroached(simplex& t, int id, Qs* q) {
  if (t.isBoundary())
    return true;
  int i;
  for (i = 0; i < 3; i++) {
    simplex tt = t.across(id, q);
    if (t.failed)
      return false;

    if (t.across().isBoundary() && (t.farAngle() > 45.0))
      break;
    t = t.rotClockwise();
  }
  if (i < 3)
    t.boundary = 1;
  return true;
  // if (i < 3)
  //  return t.boundary = 1;
  // else
  //  return 0;
}

int findAndReserveCavity(vertex* v, simplex& t, Qs* q) {
  //  t = simplex(v->badT, 0);

  if (t.t == NULL) {
    cout << "refine: nothing in badT" << endl;
    abort();
  }
  if (t.t->bad == 0)
    return 0;

  if (!t.acquire(v->id, q))
    return 2;

  // if there is an obtuse angle then move across to opposite triangle, repeat
  if (obtuse(t)) {
    simplex tt = t.across(v->id, q);
    if (t.failed)
      return 2;

    t = tt;

    if (!t.acquire(v->id, q))
      return 2;
  }
  while (t.isTriangle()) {
    int i;
    for (i = 0; i < 2; i++) {
      t = t.rotClockwise();
      if (obtuse(t)) {
        simplex tt = t.across(v->id, q);
        if (t.failed)
          return 2;
        t = tt;
        if (!t.acquire(v->id, q))
          return 2;
        break;
      }
    }
    if (i == 2)
      break;
  }

  // if encroaching on boundary, move to boundary
  if (!checkEncroached(t, v->id, q))
    return 2;

  // use circumcenter to add (if it is a boundary then its middle)
  v->pt = circumcenter(t);
  if (!reserveForInsert(v, t, v->id, q))
    return 2;
  return 1;
}

// checks if v "won" on all adjacent vertices and inserts point if so
// returns true if "won" and cavity was updated
bool addCavity(vertex* v, simplex t, Qs* q, TriangleTable TT, vertex** vbase) {
  bool flag = 1;
  for (int i = 0; i < q->vertexQ.size(); i++) {
    vertex* u = (q->vertexQ)[i];
    if (u->reserve != v->id)
      flag = 0; // someone else with higher priority reserved u
  }
  if (flag) {
    tri* t0         = t.t;
    tri* t1         = v->t; // the memory for the two new triangles
    tri* t2         = t1 + 1;
    t1->initialized = 1;
    if (t.isBoundary())
      t.splitBoundary(v, t1);
    else {
      t2->initialized = 1;
      t.split(v, t1, t2);
    }

    // update the cavity
    for (int i = 0; i < q->simplexQ.size(); i++)
      (q->simplexQ)[i].flip();
    q->simplexQ.push_back(simplex(t0, 0));
    q->simplexQ.push_back(simplex(t1, 0));
    if (!t.isBoundary())
      q->simplexQ.push_back(simplex(t2, 0));

    for (int i = 0; i < q->simplexQ.size(); i++) {
      tri* t = (q->simplexQ)[i].t;
      if (skinnyTriangle(t)) {
        // TODO, get a point v
        // TT.insert(t);
        t->bad = 1;
        if (q->begin == q->end) {
          cerr << "Ran out of new vertices\n";
          abort();
        }
        vbase[q->begin]->badT = t;
        q->newQ.push_back(q->begin);
        q->numBad++;
        q->begin++;
      } else
        t->bad = 0;
    }
    v->badT = NULL;
  }
  return flag;
}

void resetState(int id, Qs* q) {
  for (int i = 0; i < q->vertexQ.size(); i++) {
    vertex* u = (q->vertexQ)[i];
    if (u->reserve == id)
      u->reserve = -1; // reset to -1
  }
  for (int i = 0; i < q->acquireQ.size(); i++) {
    vertex* u = q->acquireQ[i];
    if (u->reserve == id)
      u->reserve = -1;
  }
  q->acquireQ.clear();
  q->simplexQ.clear();
  q->vertexQ.clear();
}

// *************************************************************
//    MAIN REFINEMENT LOOP
// *************************************************************

struct GInserter {
  vertex** vv;
  Qs** qs;
  TriangleTable TT;
  vertex** v;
  GInserter(vertex** _vv, Qs** _qs, TriangleTable _TT, vertex** _v)
      : vv(_vv), qs(_qs), TT(_TT), v(_v) {}
  void operator()(int j) {
    unsigned tid = Exp::getTID();
    Qs* q        = qs[tid];

    int cur = j;

    while (true) {
      bool success = true;
      simplex t    = simplex(vv[cur]->badT, 0);
      int r        = findAndReserveCavity(vv[cur], t, q);
      if (r == 1 && addCavity(vv[cur], t, q, TT, v)) {
        ;
      } else if (r == 2) {
        q->abortedQ.push_back(cur);
        q->aborted++;
        success = false;
      }

      resetState(vv[cur]->id, q);

      if (!success)
        break;
      if (!q->newQ.empty()) {
        cur = q->newQ.back();
        q->newQ.pop_back();
      } else if (!q->abortedQ.empty()) {
        cur = q->abortedQ.front();
        q->abortedQ.pop_front();
      } else {
        break;
      }
    }
  }
};

void addRefiningVertices(vertex** v, int n, int nTotal, TriangleTable TT,
                         int vlen, int& numBad, int& failed, int& rounds) {
  unsigned numThreads = Exp::getNumThreads();

  int maxR      = (int)numThreads;
  Qs* qqs       = newA(Qs, maxR);
  Qs** qs       = newA(Qs*, maxR);
  int blockSize = (vlen - n + numThreads - 1) / numThreads;
  for (int i = 0; i < maxR; i++) {
    qs[i]        = new (&qqs[i]) Qs;
    int b        = n + i * blockSize;
    qs[i]->begin = min(b, n + vlen);
    qs[i]->end   = min(b + blockSize, n + vlen);
    qs[i]->newQ.reserve(qs[i]->end - qs[i]->begin);
    // XXX cout << qs[i]->begin << " " << qs[i]->end << "\n";
  }
#ifdef DUMB
  simplex* t  = newA(simplex, maxR);
  bool* flags = newA(bool, n);
  vertex** h  = newA(vertex*, n);
#endif

  int top  = n;
  int size = maxR;

  // process all vertices starting just below the top
  while (top > 0) {
    int cnt     = top;
    vertex** vv = v + top - cnt;

    //    parallel_for (int j = 0; j < cnt; j++)
    parallel_doall_obj(int, j, 0, cnt, GInserter(vv, qs, TT, v)) {
      unsigned tid = Exp::getTID();
      Qs* q        = qs[tid];

      int cur = j;

      while (true) {
        bool success = true;
        simplex t    = simplex(vv[cur]->badT, 0);
        int r        = findAndReserveCavity(vv[cur], t, q);
        if (r == 1 && addCavity(vv[cur], t, q, TT, v)) {
          ;
        } else if (r == 2) {
          q->abortedQ.push_back(cur);
          q->aborted++;
          success = false;
        }

        resetState(vv[cur]->id, q);

        if (!success)
          break;
        if (!q->newQ.empty()) {
          cur = q->newQ.back();
          q->newQ.pop_back();
        } else if (!q->abortedQ.empty()) {
          cur = q->abortedQ.front();
          q->abortedQ.pop_front();
        } else {
          break;
        }
      }
    }
    parallel_doall_end

        for (int i = 0; i < numThreads; i++) {
      Qs* q = qs[i];
      for (int j = 0; j < q->abortedQ.size(); ++j) {
        int cur   = q->abortedQ[j];
        simplex t = simplex(vv[cur]->badT, 0);
        int r     = findAndReserveCavity(vv[cur], t, q);
        if (r == 1 && addCavity(vv[cur], t, q, TT, v)) {
          ;
        } else if (r == 2) {
          abort();
        }
        resetState(vv[cur]->id, q);
      }

      failed += q->aborted;
      q->abortedQ.clear();
      q->aborted = 0;

      while (!q->newQ.empty()) {
        int cur = q->newQ.back();
        q->newQ.pop_back();

        simplex t = simplex(vv[cur]->badT, 0);
        int r     = findAndReserveCavity(vv[cur], t, q);
        if (r == 1 && addCavity(vv[cur], t, q, TT, v)) {
          ;
        } else if (r == 2) {
          abort();
        }
        resetState(vv[cur]->id, q);
      }

      numBad += q->numBad;
      q->newQ.clear();
      q->numBad = 0;
    }

#ifdef DUMB
    // Pack the failed vertices back onto Q
    int k = sequence::pack(vv, h, flags, cnt);
    //    parallel_for (int j = 0; j < k; j++) vv[j] = h[j];
    parallel_doall(int, j, 0, k) { vv[j] = h[j]; }
    parallel_doall_end failed += k;
#else
    int k = 0;
#endif
    top = top - cnt + k; // adjust top, accounting for failed vertices
    ++rounds;
  }
  free(qqs);
  free(qs);
#ifdef DUMB
  free(t);
  free(flags);
  free(h);
#endif
}

// *************************************************************
//    DRIVER
// *************************************************************

namespace {
struct GFn1 {
  tri* Triangs;
  GFn1(tri* _Triangs, int dummy) : Triangs(_Triangs) {}
  void operator()(int i) {
    Triangs[i].id          = i;
    Triangs[i].initialized = 0;
  }
};

struct GFn2 {
  vertex** v;
  vertex* vv;
  int n;
  tri* Triangs;
  int m;
  GFn2(vertex** _v, vertex* _vv, int _n, tri* _Triangs, int _m)
      : v(_v), vv(_vv), n(_n), Triangs(_Triangs), m(_m) {}
  void operator()(int i) {
    v[i] = new (&vv[i + n]) vertex(point2d(0, 0), i + n);
    // give each one a pointer to two triangles to use
    v[i]->t = Triangs + m + 2 * i;
  }
};

struct GFn3 {
  tri* Triangs;
  TriangleTable& workQ;
  GFn3(tri* _Triangs, TriangleTable& _workQ)
      : Triangs(_Triangs), workQ(_workQ) {}
  void operator()(int i) {
    if (skinnyTriangle(&Triangs[i])) {
      workQ.insert(&Triangs[i]);
      Triangs[i].bad = 1;
    }
  }
};

struct GFn4 {
  bool* flags;
  _seq<tri*>& badTT;
  GFn4(bool* _flags, _seq<tri*>& _badTT) : flags(_flags), badTT(_badTT) {}
  void operator()(int i) { flags[i] = badTT.A[i]->bad; }
};

struct GFn5 {
  _seq<tri*>& badT;
  vertex** v;
  int numPoints;
  int n;
  GFn5(_seq<tri*>& _badT, vertex** _v, int _numPoints, int _n)
      : badT(_badT), v(_v), numPoints(_numPoints), n(_n) {}
  void operator()(int i) {
    badT.A[i]->bad             = 2; // used to detect whether touched
    v[i + numPoints - n]->badT = badT.A[i];
  }
};

struct GFn6 {
  _seq<tri*>& badT;
  TriangleTable& workQ;
  GFn6(_seq<tri*>& _badT, TriangleTable& _workQ) : badT(_badT), workQ(_workQ) {}
  void operator()(int i) {
    if (badT.A[i]->bad == 2)
      workQ.insert(badT.A[i]);
  }
};

struct GFn7 {
  bool* flag;
  vertex* vv;
  GFn7(bool* _flag, vertex* _vv) : flag(_flag), vv(_vv) {}
  void operator()(int i) { flag[i] = (vv[i].badT == NULL); }
};

struct GFn8 {
  vertex* vv;
  int* II;
  point2d* rp;
  GFn8(vertex* _vv, int* _II, point2d* _rp) : vv(_vv), II(_II), rp(_rp) {}
  void operator()(int i) {
    vv[II[i]].id = i;
    rp[i]        = vv[II[i]].pt;
  }
};

struct GFn9 {
  bool* flag;
  tri* Triangs;
  GFn9(bool* _flag, tri* _Triangs) : flag(_flag), Triangs(_Triangs) {}
  void operator()(int i) { flag[i] = Triangs[i].initialized; }
};

struct GFn10 {
  tri* Triangs;
  _seq<int> I;
  triangle* rt;
  GFn10(tri* _Triangs, _seq<int>& _I, triangle* _rt)
      : Triangs(_Triangs), I(_I), rt(_rt) {}
  void operator()(int i) {
    tri t = Triangs[I.A[i]];
    rt[i] = triangle(t.vtx[0]->id, t.vtx[1]->id, t.vtx[2]->id);
  }
};
} // namespace

triangles<point2d> refine(triangles<point2d> Tri) {
  // following line is used to fool icpc into starting the scheduler
  if (Tri.numPoints < 0)
    cilk_spawn printf("ouch");
  startTime();
  int expandFactor   = 4;
  int n              = Tri.numPoints;
  int m              = Tri.numTriangles;
  int extraVertices  = expandFactor * n;
  int totalVertices  = n + extraVertices;
  int totalTriangles = m + 2 * extraVertices;

  vertex** v   = newA(vertex*, extraVertices);
  vertex* vv   = newA(vertex, totalVertices);
  tri* Triangs = newA(tri, totalTriangles);
  topologyFromTriangles(Tri, &vv, &Triangs);
  nextTime("from Triangles");

  //  set up extra triangles
  //  parallel_for (int i=m; i < totalTriangles; i++) {
  parallel_doall_obj(int, i, m, totalTriangles, GFn1(Triangs, 1)) {
    Triangs[i].id          = i;
    Triangs[i].initialized = 0;
  }
  parallel_doall_end

  //  set up extra vertices
  //  parallel_for (int i=0; i < totalVertices-n; i++) {
  parallel_doall_obj(int, i, 0, totalVertices - n, GFn2(v, vv, n, Triangs, m)) {
    v[i] = new (&vv[i + n]) vertex(point2d(0, 0), i + n);
    // give each one a pointer to two triangles to use
    v[i]->t = Triangs + m + 2 * i;
  }
  parallel_doall_end nextTime("initializing");

  // these will increase as more are added
  int numTriangs = m;
  int numPoints  = n;

  TriangleTable workQ = makeTriangleTable(numTriangs);
  //  parallel_for(int i=0; i < numTriangs; i++) {
  parallel_doall_obj(int, i, 0, numTriangs, GFn3(Triangs, workQ)) {
    if (skinnyTriangle(&Triangs[i])) {
      workQ.insert(&Triangs[i]);
      Triangs[i].bad = 1;
    }
  }
  parallel_doall_end nextTime("Start");

  int failed = 0;
  int rounds = 0;

  // Each iteration processes all bad triangles from the workQ while
  // adding new bad triangles to a new queue
  while (1) {
    _seq<tri*> badTT = workQ.entries();
    workQ.del();

    // packs out triangles that are no longer bad
    bool* flags = newA(bool, badTT.n);
    //    parallel_for (int i=0; i < badTT.n; i++)
    parallel_doall_obj(int, i, 0, badTT.n, GFn4(flags, badTT)) {
      flags[i] = badTT.A[i]->bad;
    }
    parallel_doall_end _seq<tri*> badT =
        sequence::pack(badTT.A, flags, badTT.n);
    free(flags);
    badTT.del();
    int numBad = badT.n;

    // XXX cout << "numBad = " << numBad << endl;
    if (numBad == 0)
      break;
    if (numPoints + numBad > totalVertices) {
      cout << "ran out of vertices" << endl;
      abort();
    }

    // allocate 1 vertex per bad triangle and assign triangle to it
    //    parallel_for (int i=0; i < numBad; i++) {
    parallel_doall_obj(int, i, 0, numBad, GFn5(badT, v, numPoints, n)) {
      badT.A[i]->bad             = 2; // used to detect whether touched
      v[i + numPoints - n]->badT = badT.A[i];
    }
    parallel_doall_end

        // the new work queue
        workQ = makeTriangleTable(numBad);

    // This does all the work
    addRefiningVertices(v + numPoints - n, numBad, numPoints, workQ,
                        extraVertices - numPoints + n, numBad, failed, rounds);

#ifdef DUMB
    // push any bad triangles that were left untouched onto the Q
    //    parallel_for (int i=0; i < numBad; i++)
    parallel_doall_obj(int, i, 0, numBad, GFn6(badT, workQ)) {
      if (badT.A[i]->bad == 2)
        workQ.insert(badT.A[i]);
    }
    parallel_doall_end badT.del();

    numPoints += numBad;
    numTriangs += 2 * numBad;
#else
    numPoints += numBad;
    numTriangs += 2 * numBad;

    break;
#endif
  }

  cout << "failed = " << failed << "\n";
  cout << "rounds = " << rounds << "\n";

  nextTime("refinement");

  // Extract Vertices for result
  bool* flag = newA(bool, numTriangs);
  //  parallel_for (int i=0; i < numPoints; i++) {
  parallel_doall_obj(int, i, 0, numPoints, GFn7(flag, vv)) {
    flag[i] = (vv[i].badT == NULL);
  }
  parallel_doall_end _seq<int> I = sequence::packIndex(flag, numPoints);
  int nO                         = I.n;
  int* II                        = I.A;
  point2d* rp                    = newA(point2d, nO);
  //  parallel_for (int i=0; i < nO; i++) {
  parallel_doall_obj(int, i, 0, nO, GFn8(vv, II, rp)) {
    vv[II[i]].id = i;
    rp[i]        = vv[II[i]].pt;
  }
  parallel_doall_end cout << "total points = " << nO << endl;
  I.del();

  // Extract Triangles for result
  //  parallel_for (int i=0; i < numTriangs; i++)
  parallel_doall_obj(int, i, 0, numTriangs, GFn9(flag, Triangs)) {
    flag[i] = Triangs[i].initialized;
  }
  parallel_doall_end I = sequence::packIndex(flag, numTriangs);
  triangle* rt         = newA(triangle, I.n);
  //  parallel_for (int i=0; i < I.n; i++) {
  parallel_doall_obj(int, i, 0, I.n, GFn10(Triangs, I, rt)) {
    tri t = Triangs[I.A[i]];
    rt[i] = triangle(t.vtx[0]->id, t.vtx[1]->id, t.vtx[2]->id);
  }
  parallel_doall_end cout << "total triangles = " << I.n << endl;

  I.del();
  free(flag);
  free(Triangs);
  free(v);
  free(vv);
  nextTime("finish");
  return triangles<point2d>(nO, I.n, rp, rt);
}
