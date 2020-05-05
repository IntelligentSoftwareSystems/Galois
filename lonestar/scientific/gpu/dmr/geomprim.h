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
#pragma once

__device__
FORD distanceSquare(FORD onex, FORD oney, FORD twox, FORD twoy) {
  FORD dx = onex - twox;
  FORD dy = oney - twoy;
  FORD dsq = dx * dx + dy * dy;
  return dsq;
}

__device__
FORD distanceSquare(unsigned one, unsigned two, FORD *nodex, FORD *nodey) {
  return distanceSquare(nodex[one], nodey[one], nodex[two], nodey[two]);
}

__device__ bool angleOB(Mesh &mesh, unsigned a, unsigned b, unsigned c)
{
  FORD vax = mesh.nodex[a] - mesh.nodex[c];
  FORD vay = mesh.nodey[a] - mesh.nodey[c];
  FORD vbx = mesh.nodex[b] - mesh.nodex[c];
  FORD vby = mesh.nodey[b] - mesh.nodey[c];
  FORD dp = vax * vbx + vay * vby; // dot-product?

  if (dp < 0) 
    return true;

  return false;
}
__device__ bool angleLT(Mesh &mesh, unsigned a, unsigned b, unsigned c) {
  FORD vax = mesh.nodex[a] - mesh.nodex[c];
  FORD vay = mesh.nodey[a] - mesh.nodey[c];
  FORD vbx = mesh.nodex[b] - mesh.nodex[c];
  FORD vby = mesh.nodey[b] - mesh.nodey[c];
  FORD dp = vax * vbx + vay * vby; // dot-product?

  if (dp < 0) {
    // id is obtuse at point ii.    
    return false;
  } else {
    FORD dsqaacurr = distanceSquare(a, c, mesh.nodex, mesh.nodey);
    FORD dsqbbcurr = distanceSquare(b, c, mesh.nodex, mesh.nodey);
    FORD c = dp * rsqrtf(dsqaacurr * dsqbbcurr);
    if (c > cos(MINANGLE * (PI / 180))) {
      return true;
    }
  }

  return false;
}

// > 0, in circle, == 0, on circle, < 0, outside circle
// assumes a, b, c are in counter-clockwise order
// code from triangle
__device__
FORD gincircle (FORD ax, FORD ay, FORD bx, FORD by, FORD cx, FORD cy, FORD px, FORD py) {
  FORD apx, bpx, cpx, apy, bpy, cpy;
  FORD bpxcpy, cpxbpy, cpxapy, apxcpy, apxbpy, bpxapy;
  FORD alift, blift, clift, det;

  apx = ax - px;
  bpx = bx - px;
  cpx = cx - px;
  
  apy = ay - py;
  bpy = by - py;
  cpy = cy - py;
  
  bpxcpy = bpx * cpy;
  cpxbpy = cpx * bpy;
  alift = apx * apx + apy * apy;
  
  cpxapy = cpx * apy;
  apxcpy = apx * cpy;
  blift = bpx * bpx + bpy * bpy;
  
  apxbpy = apx * bpy;
  bpxapy = bpx * apy;
  clift = cpx * cpx + cpy * cpy;
  
  det = alift * (bpxcpy - cpxbpy) + blift * (cpxapy - apxcpy) + clift * (apxbpy - bpxapy);
  
  return det;
}

__device__
FORD counterclockwise(FORD pax, FORD pay, FORD pbx, FORD pby, FORD pcx, FORD pcy)
{
  FORD detleft, detright, det;

  detleft = (pax - pcx) * (pby - pcy);
  detright = (pay - pcy) * (pbx - pcx);
  det = detleft - detright;

  return det;
}

__device__
void circumcenter(FORD Ax, FORD Ay, FORD Bx, FORD By, FORD Cx, FORD Cy, FORD &CCx, FORD &CCy) {
  FORD D;

  D = 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By));

  CCx = ((Ax*Ax + Ay*Ay)*(By - Cy) + (Bx*Bx + By*By)*(Cy - Ay) + (Cx*Cx + Cy*Cy)*(Ay - By))/D;
  CCy = ((Ax*Ax + Ay*Ay)*(Cx - Bx) + (Bx*Bx + By*By)*(Ax - Cx) + (Cx*Cx + Cy*Cy)*(Bx - Ax))/D;
}

