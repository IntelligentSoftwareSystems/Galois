#include "geometry.h"
//#include "kdTree.h"

// There are 3 versions in here
// The second is definitely slower than the first
// The third is broken into two parts, where the first only depends 
//   on the triangle (slow) and the second adds the ray

#define EPSILON 0.00000001

// Code is based on:
// Fast, Minimum Storage Ray/Triangle Intersection
// Tomas Moller and Ben Trumbore
template <class floatT>
inline floatT rayTriangleIntersect(ray<_point3d<floatT> > R, 
				   _point3d<floatT> m[]) {
  typedef _point3d<floatT> pointT;
  typedef _vect3d<floatT> vectT;
  pointT o = R.o;
  vectT d = R.d;
  vectT e1 = m[1] - m[0];
  vectT e2 = m[2] - m[0];

  vectT pvec = d.cross(e2);
  floatT det = e1.dot(pvec);

  // if determinant is zero then ray is
  // parallel with the triangle plane
  if (det > -EPSILON && det < EPSILON) return 0;
  floatT invDet = 1.0/det;

  // calculate distance from m[0] to origin
  vectT tvec = o - m[0];

  // u and v are the barycentric coordinates
  // in triangle if u >= 0, v >= 0 and u + v <= 1
  floatT u = tvec.dot(pvec) * invDet;

  // check against one edge and opposite point
  if (u < 0.0 || u > 1.0) return 0;

  vectT qvec = tvec.cross(e1);
  floatT v = d.dot(qvec) * invDet;

  // check against other edges
  if (v < 0.0 || u + v > 1.0) return 0;

  //distance along the ray, i.e. intersect at o + t * d
  floatT t = e2.dot(qvec) * invDet;
  
  return t;
}
/*

// Copyright 2001, softSurfer (www.softsurfer.com)
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.
//===================================================================

// intersect_RayTriangle(): intersect a ray with a 3D triangle
//    Input:  a ray R, and a triangle T
//    Output: *I = intersection point (when it exists)
//    Return: -1 = triangle is degenerate (a segment or point)
//             0 = disjoint (no intersect)
//             1 = intersect in unique point I1
//             2 = are in the same plane
inline floatT rayTriangleIntersect2(ray R, point3d T[] )
{
  // get triangle edge vectors and plane normal
  vect3d u = T[1] - T[0];
  vect3d v = T[2] - T[0];
  vect3d n = u.cross(v);
  //if (n == (Vector) 0) return -1;         // triangle is degenerate

  vect3d w0 = R.o - T[0];
  floatT a = -n.dot(w0);
  floatT b = n.dot(R.d);
  if (fabs(b) < EPSILON) {     // ray is parallel to triangle plane
    if (a == 0) return 0;        // ray lies in triangle plane
    else return 0;             // ray disjoint from plane
  }

  // get intersect point of ray with triangle plane
  floatT r = a / b;
  if (r < 0.0) return 0;         // ray goes away from triangle
  point3d I = R.o + R.d * r;     // intersect point of ray and plane

  // is I inside T?
  floatT uu = u.dot(u);
  floatT uv = u.dot(v);
  floatT vv = v.dot(v);
  vect3d w = I - T[0];
  floatT wu = w.dot(u);
  floatT wv = w.dot(v);
  floatT D = uv * uv - uu * vv;
  floatT invD = 1.0/D;

  // get and test parametric coords
  floatT s = (uv * wv - vv * wu) * invD;
  //cout << "u=" << s << endl;
  if (s < 0.0 || s > 1.0)        // I is outside T
    return 0;
  floatT t = (uv * wu - uu * wv) * invD;
  //cout << "v=" << t << endl;

  if (t < 0.0 || (s + t) > 1.0)  // I is outside T
    return 0;

  return r;                      // I is in T
}

// A more efficient version if going to use the same triangle many times
// although it takes more space and including preprocessing takes more time
// It also seems to return true in some "near cases" in which I believe
// false is correct.   This might be OK for some applications
  
int invert3x3(vect3d *m, vect3d *mInv) {
  vect3d c12 = m[1].cross(m[2]);
  floatT determinant = m[0].dot(c12);
  if (determinant > -EPSILON && determinant < EPSILON) return 0;
  floatT dinv = 1.0/determinant;
  mInv[0] = c12 * dinv;
  mInv[1] = m[2].cross(m[0]) * dinv;
  mInv[2] = m[0].cross(m[1]) * dinv;
  return 1;
}

struct triInfo2 {
  point3d p0;
  vect3d bInv[3];

  triInfo2() {}

  // could save time by reusing subresults
  triInfo2(point3d* T) {
    vect3d b[3];
    p0 = T[0];
    b[0] = T[1]-p0;
    b[1] = T[2]-p0;
    vect3d c = b[0].cross(b[1]);
    b[2] = c / c.Length();
    invert3x3(b,bInv);
  }

  inline floatT intersectRay(ray R) {
    point3d o = R.o;
    vect3d d = R.d;
    floatT den = d.dot(bInv[2]);
    if (den > -EPSILON && den < EPSILON) return 0;
    floatT num = (p0 - o).dot(bInv[2]);
    floatT t = num/den;
    if (t <= 0) return 0;
    point3d ip = o + d * t;       // intersection point on plane
    vect3d p = ip - p0;           // relative to triangle corner
    floatT u = p.dot(bInv[0]);
    floatT v = p.dot(bInv[1]);
    if (u < 0.0 || v < 0.0 || u + v > 1.0) return 0;
    return t;
  }
};

struct triInfo {
  point3d p0;
  vect3d b0;
  vect3d b1;
  vect3d b2;
  triInfo(point3d _p0, vect3d _b0, vect3d _b1, vect3d _b2) :
    p0(_p0), b0(_b0), b1(_b1), b2(_b2) {}
};

  // could save time by reusing subresults
  triInfo tInfo(point3d* T) {
    vect3d b[3];
    vect3d bInv[3];
    point3d p0 = T[0];
    b[0] = T[1]-p0;
    b[1] = T[2]-p0;
    vect3d c = b[0].cross(b[1]);
    b[2] = c / c.Length();
    invert3x3(b,bInv);
    return triInfo(p0, bInv[0], bInv[1], bInv[2]);
  }

inline floatT intersectRay(triInfo Ti, ray R) {
    point3d o = R.o;
    vect3d d = R.d;
    floatT den = d.dot(Ti.b2);
    if (den > -EPSILON && den < EPSILON) return 0;
    floatT num = (Ti.p0 - o).dot(Ti.b2);
    floatT t = num/den;
    //if (t <= 0) return 0;
    point3d ip = o + d * t;       // intersection point on plane
    vect3d p = ip - Ti.p0;           // relative to triangle corner
    floatT u = p.dot(Ti.b0);
    if (u < 0.0 || u > 1.0) return 0;
    floatT v = p.dot(Ti.b1);
    if (v < 0.0 || u + v > 1.0) return 0;
    return t;
  }

//floatT rayTriangleIntersect3(ray R, point3d m[]) {
//  triInfo ti(m);
//  return ti.intersectRay(R);
//}
*/
