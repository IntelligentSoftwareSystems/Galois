/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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
#include "sharedptr.h"

#define MINANGLE 30
#define PI 3.14159265358979323846 // from C99 standard.
#define FORD double
#define DIMSTYPE unsigned

#define INVALIDID 1234567890
#define MAXID INVALIDID

// "usual" ratio of final nodes to final elements, determined empirically
// used to adjust maxfactor for nodes
// use 1 to be conservative
#define MAX_NNODES_TO_NELEMENTS 2

struct ShMesh {
  uint maxnelements;
  uint maxnnodes;
  uint ntriangles;
  uint nnodes;
  uint nsegments;
  uint nelements;

  Shared<FORD> nodex;
  Shared<FORD> nodey;
  Shared<uint3> elements;
  Shared<uint3> neighbours;
  Shared<bool> isdel;
  Shared<bool> isbad;
  Shared<int> owners;
};

struct Mesh {
  uint maxnelements;
  uint maxnnodes;
  uint ntriangles;
  uint nnodes;
  uint nsegments;
  uint nelements;

  FORD* nodex; // could be combined
  FORD* nodey;
  uint3* elements;
  volatile bool* isdel;
  bool* isbad;
  uint3* neighbours;
  int* owners;

  Mesh() {}

  Mesh(ShMesh& mesh) {
    maxnelements = mesh.maxnelements;
    maxnnodes    = mesh.maxnnodes;
    ntriangles   = mesh.ntriangles;
    nnodes       = mesh.nnodes;
    nsegments    = mesh.nsegments;
    nelements    = mesh.nelements;

    nodex      = mesh.nodex.gpu_wr_ptr();
    nodey      = mesh.nodey.gpu_wr_ptr();
    elements   = mesh.elements.gpu_wr_ptr();
    neighbours = mesh.neighbours.gpu_wr_ptr();
    isdel      = mesh.isdel.gpu_wr_ptr();
    isbad      = mesh.isbad.gpu_wr_ptr();
    owners     = mesh.owners.gpu_wr_ptr(true);
  }

  void refresh(ShMesh& mesh) {
    maxnelements = mesh.maxnelements;
    maxnnodes    = mesh.maxnnodes;
    ntriangles   = mesh.ntriangles;
    nnodes       = mesh.nnodes;
    nsegments    = mesh.nsegments;
    nelements    = mesh.nelements;

    nodex      = mesh.nodex.gpu_wr_ptr();
    nodey      = mesh.nodey.gpu_wr_ptr();
    elements   = mesh.elements.gpu_wr_ptr();
    neighbours = mesh.neighbours.gpu_wr_ptr();
    isdel      = mesh.isdel.gpu_wr_ptr();
    isbad      = mesh.isbad.gpu_wr_ptr();
    owners     = mesh.owners.gpu_wr_ptr(true);
  }
};

#define IS_SEGMENT(element) (((element).z == INVALIDID))
