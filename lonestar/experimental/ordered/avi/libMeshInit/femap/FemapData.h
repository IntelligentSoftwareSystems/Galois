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

/**
 * FemapData.h: Contains data structures that store data from Femap Neutral file
 * format DG++
 *
 * Copyright (c) 2006 Adrian Lew
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "AuxDefs.h"

struct femapMaterial {
  size_t id;
  size_t type;
  size_t subtype;
  std::string title;
  int bval[10];
  int ival[25];
  double mval[200];
};

struct femapProperty {
  size_t id;
  size_t matId;
  size_t type;
  std::string title;
  int flag[4];
  int num_val;
  VecDouble value;
};

struct femapNode {
  size_t id;
  //! coordinates
  double x[3];
  int permBc[6];
};

struct femapElement {
  size_t id;
  size_t propId;
  size_t type;
  size_t topology;
  size_t geomId;
  int formulation; // int is a guess--documentation doesn't give type
  VecSize_t node;
};

struct constraint {
  size_t id;
  bool dof[6];
  int ex_geom;
};

struct femapConstraintSet {
  size_t id;
  std::string title;
  std::vector<constraint> nodalConstraint;
};

struct load {
  size_t id;
  size_t type;
  int dof_face[3];
  double value[3];
  bool is_expanded;
};

struct femapLoadSet {
  size_t id;
  std::string title;
  double defTemp;
  bool tempOn;
  bool gravOn;
  bool omegaOn;
  double grav[6];
  double origin[3];
  double omega[3];
  std::vector<load> loads;
};

struct groupRule {
  size_t type;
  size_t startID;
  size_t stopID;
  size_t incID;
  size_t include;
};

struct groupList {
  size_t type;
  VecSize_t entityID;
};

struct femapGroup {
  size_t id;
  short int need_eval;
  std::string title;
  int layer[2];
  int layer_method;
  std::vector<groupRule> rules;
  std::vector<groupList> lists;
};
