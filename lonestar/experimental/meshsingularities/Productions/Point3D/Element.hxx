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

#ifndef __ELEMENT_3D_H_INCLUDED__
#define __ELEMENT_3D_H_INCLUDED__

#include "EPosition.hxx"
#include "TripleArgFunction.hxx"
#include "GaussianQuadrature.hxx"
#include "NPosition.hxx"
#include <cstdlib>
#include <cmath>
#include <map>
namespace D3 {

class TripleArgFunctionProduct : public ITripleArgFunction {

private:
  ITripleArgFunction* function1;
  ITripleArgFunction* function2;

public:
  void SetFunctions(ITripleArgFunction* _function1,
                    ITripleArgFunction* _function2) {
    function1 = _function1;
    function2 = _function2;
  }

  virtual double ComputeValue(double x, double y, double z) {
    return function1->ComputeValue(x, y, z) * function2->ComputeValue(x, y, z);
  }
};

class Element {

private:
  double xl;
  double yl;
  double xr;
  double yr;
  double zl;
  double zr;
  double size;
  bool* neighbours;
  EPosition position;

  static const int nr_of_nodes = 27;

  ITripleArgFunction** shapeFunctions;
  int* shapeFunctionNrs;
  TripleArgFunctionProduct* product;

  void SetIternalBotInterfaceNumbers(int nr, Element* bot_left_near_element,
                                     Element* bot_left_far_element,
                                     Element* bot_right_far_element,
                                     Element* top_left_near_element,
                                     Element* top_left_far_element,
                                     Element* top_right_far_elemnt,
                                     Element* top_right_near_element);

public:
  static const int vertex_bot_left_near  = 0;
  static const int edge_bot_left         = 1;
  static const int vertex_bot_left_far   = 2;
  static const int edge_bot_far          = 3;
  static const int vertex_bot_right_far  = 4;
  static const int edge_left_near        = 5;
  static const int face_left             = 6;
  static const int edge_left_far         = 7;
  static const int face_far              = 8;
  static const int edge_right_far        = 9;
  static const int vertex_top_left_near  = 10;
  static const int edge_top_left         = 11;
  static const int vertex_top_left_far   = 12;
  static const int edge_top_far          = 13;
  static const int vertex_top_right_far  = 14;
  static const int edge_bot_near         = 15;
  static const int face_bot              = 16;
  static const int edge_bot_right        = 17;
  static const int face_near             = 18;
  static const int interior              = 19;
  static const int face_right            = 20;
  static const int edge_top_near         = 21;
  static const int face_top              = 22;
  static const int edge_top_right        = 23;
  static const int vertex_bot_right_near = 24;
  static const int edge_right_near       = 25;
  static const int vertex_top_right_near = 26;

  Element(double xl, double yl, double zl, double size, bool* _neighbours,
          EPosition position)
      : xl(xl), yl(yl), zl(zl), size(size), position(position) {

    xr         = xl + size;
    yr         = yl + size;
    zr         = zl + size;
    neighbours = new bool[18];
    for (int i = 0; i < 18; i++)
      neighbours[i] = _neighbours[i];
    double coordinates[6];
    coordinates[0]   = xl;
    coordinates[1]   = xr;
    coordinates[2]   = yl;
    coordinates[3]   = yr;
    coordinates[4]   = zl;
    coordinates[5]   = zr;
    shapeFunctions   = new ITripleArgFunction*[nr_of_nodes];
    shapeFunctionNrs = new int[nr_of_nodes];

    shapeFunctions[vertex_bot_left_near] =
        new VertexBotLeftNearShapeFunction(coordinates, neighbours, position);
    shapeFunctions[vertex_bot_left_far] =
        new VertexBotLeftFarShapeFunction(coordinates, neighbours, position);
    shapeFunctions[vertex_bot_right_near] =
        new VertexBotRightNearShapeFunction(coordinates, neighbours, position);
    shapeFunctions[vertex_bot_right_far] =
        new VertexBotRightFarShapeFunction(coordinates, neighbours, position);
    shapeFunctions[vertex_top_left_near] =
        new VertexTopLeftNearShapeFunction(coordinates, neighbours, position);
    shapeFunctions[vertex_top_left_far] =
        new VertexTopLeftFarShapeFunction(coordinates, neighbours, position);
    shapeFunctions[vertex_top_right_near] =
        new VertexTopRightNearShapeFunction(coordinates, neighbours, position);
    shapeFunctions[vertex_top_right_far] =
        new VertexTopRightFarShapeFunction(coordinates, neighbours, position);

    shapeFunctions[edge_bot_left] =
        new EdgeBotLeftShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_bot_right] =
        new EdgeBotRightShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_bot_near] =
        new EdgeBotNearShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_bot_far] =
        new EdgeBotFarShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_top_left] =
        new EdgeTopLeftShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_top_right] =
        new EdgeTopRightShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_top_near] =
        new EdgeTopNearShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_top_far] =
        new EdgeTopFarShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_left_near] =
        new EdgeLeftNearShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_left_far] =
        new EdgeLeftFarShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_right_near] =
        new EdgeRightNearShapeFunction(coordinates, neighbours, position);
    shapeFunctions[edge_right_far] =
        new EdgeRightFarShapeFunction(coordinates, neighbours, position);

    shapeFunctions[face_left] =
        new FaceLeftShapeFunction(coordinates, neighbours, position);
    shapeFunctions[face_right] =
        new FaceRightShapeFunction(coordinates, neighbours, position);
    shapeFunctions[face_top] =
        new FaceTopShapeFunction(coordinates, neighbours, position);
    shapeFunctions[face_bot] =
        new FaceBotShapeFunction(coordinates, neighbours, position);
    shapeFunctions[face_near] =
        new FaceNearShapeFunction(coordinates, neighbours, position);
    shapeFunctions[face_far] =
        new FaceFarShapeFunction(coordinates, neighbours, position);

    shapeFunctions[interior] =
        new InteriorShapeFunction(coordinates, neighbours, position);
    product = new TripleArgFunctionProduct();
  }

  ~Element() {
    delete[] shapeFunctionNrs;
    for (int i = 0; i < nr_of_nodes; i++)
      delete shapeFunctions[i];

    delete[] shapeFunctions;
    delete[] neighbours;
    delete product;
  }

  Element** CreateAnotherTier(int nr);
  Element** CreateFirstTier(int nr);
  Element** CreateLastTier(int nr);

  void fillMatrix(double** tier_matrix, double** global_matrix,
                  int start_nr_adj);
  void fillRhs(double* tier_rhs, double* global_rhs, ITripleArgFunction* f,
               int start_nr_adj);
  void fillMatrices(double** tier_matrix, double** global_matrix,
                    double* tier_rhs, double* global_rhs, ITripleArgFunction* f,
                    int start_nr_adj);
  bool checkSolution(std::map<int, double>* solution_map,
                     ITripleArgFunction* f);

  void set_node_nr(int node, int node_nr) { shapeFunctionNrs[node] = node_nr; }
  int get_node_nr(int node) { return shapeFunctionNrs[node]; }
  int get_bot_left_near_vertex_nr() {
    return shapeFunctionNrs[vertex_bot_left_near];
  }

private:
  void comp(int indx1, int indx2, ITripleArgFunction* f1,
            ITripleArgFunction* f2, double** tier_matrix,
            double** global_matrix, int start_nr_adj);
};
} // namespace D3

#endif
