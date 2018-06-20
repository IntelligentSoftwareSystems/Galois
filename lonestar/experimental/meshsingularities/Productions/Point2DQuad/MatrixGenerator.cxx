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

#include "MatrixGenerator.hxx"

using namespace D2Quad;

void set_big_interface_nrs(int nr, bool first_tier, Element** elements) {
  int start_nr = nr;
  // left edge
  elements[0]->set_bot_left_vertex_nr(nr++);
  elements[0]->set_left_edge_nr(nr++);
  elements[0]->set_top_left_vertex_nr(nr);
  if (!first_tier)
    nr -= 2;
  elements[1]->set_bot_left_vertex_nr(nr++);
  elements[1]->set_left_edge_nr(nr++);
  elements[1]->set_top_left_vertex_nr(nr);
  elements[2]->set_bot_left_vertex_nr(nr++);
  elements[2]->set_left_edge_nr(nr++);
  elements[2]->set_top_left_vertex_nr(nr);
  if (!first_tier)
    nr -= 2;
  elements[3]->set_bot_left_vertex_nr(nr++);
  elements[3]->set_left_edge_nr(nr++);
  elements[3]->set_top_left_vertex_nr(nr);

  // top edge
  elements[3]->set_top_left_vertex_nr(nr++);
  elements[3]->set_top_edge_nr(nr++);
  elements[3]->set_top_right_vertex_nr(nr);
  if (!first_tier)
    nr -= 2;
  elements[4]->set_top_left_vertex_nr(nr++);
  elements[4]->set_top_edge_nr(nr++);
  elements[4]->set_top_right_vertex_nr(nr);
  elements[5]->set_top_left_vertex_nr(nr++);
  elements[5]->set_top_edge_nr(nr++);
  elements[5]->set_top_right_vertex_nr(nr);
  if (!first_tier)
    nr -= 2;
  elements[6]->set_top_left_vertex_nr(nr++);
  elements[6]->set_top_edge_nr(nr++);
  elements[6]->set_top_right_vertex_nr(nr);

  // right edge
  elements[6]->set_top_right_vertex_nr(nr++);
  elements[6]->set_right_edge_nr(nr++);
  elements[6]->set_bot_right_vertex_nr(nr);
  if (!first_tier)
    nr -= 2;
  elements[7]->set_top_right_vertex_nr(nr++);
  elements[7]->set_right_edge_nr(nr++);
  elements[7]->set_bot_right_vertex_nr(nr);
  elements[8]->set_top_right_vertex_nr(nr++);
  elements[8]->set_right_edge_nr(nr++);
  elements[8]->set_bot_right_vertex_nr(nr);
  if (!first_tier)
    nr -= 2;
  elements[9]->set_top_right_vertex_nr(nr++);
  elements[9]->set_right_edge_nr(nr++);
  elements[9]->set_bot_right_vertex_nr(nr);

  // bot edge
  elements[9]->set_bot_right_vertex_nr(nr++);
  elements[9]->set_bot_edge_nr(nr++);
  elements[9]->set_bot_left_vertex_nr(nr);
  if (!first_tier)
    nr -= 2;
  elements[10]->set_bot_right_vertex_nr(nr++);
  elements[10]->set_bot_edge_nr(nr++);
  elements[10]->set_bot_left_vertex_nr(nr);
  elements[11]->set_bot_right_vertex_nr(nr++);
  elements[11]->set_bot_edge_nr(nr++);
  if (!first_tier)
    elements[11]->set_bot_left_vertex_nr(start_nr);
  else
    elements[11]->set_bot_left_vertex_nr(nr);
  if (!first_tier)
    nr -= 2;
  elements[0]->set_bot_right_vertex_nr(nr++);
  elements[0]->set_bot_edge_nr(nr);
}

void set_interior_nrs(int nr, Element** elements) {
  elements[0]->set_interior_nr(nr++);
  elements[0]->set_top_edge_nr(nr);
  elements[1]->set_bot_edge_nr(nr++);
  elements[1]->set_interior_nr(nr++);
  elements[1]->set_top_edge_nr(nr);
  elements[2]->set_bot_edge_nr(nr++);
  elements[2]->set_interior_nr(nr++);
  elements[2]->set_top_edge_nr(nr);
  elements[3]->set_bot_edge_nr(nr++);
  elements[3]->set_interior_nr(nr++);
  elements[3]->set_right_edge_nr(nr);
  elements[4]->set_left_edge_nr(nr++);
  elements[4]->set_interior_nr(nr++);
  elements[4]->set_right_edge_nr(nr);
  elements[5]->set_left_edge_nr(nr++);
  elements[5]->set_interior_nr(nr++);
  elements[5]->set_right_edge_nr(nr);
  elements[6]->set_left_edge_nr(nr++);
  elements[6]->set_interior_nr(nr++);
  elements[6]->set_bot_edge_nr(nr);
  elements[7]->set_top_edge_nr(nr++);
  elements[7]->set_interior_nr(nr++);
  elements[7]->set_bot_edge_nr(nr);
  elements[8]->set_top_edge_nr(nr++);
  elements[8]->set_interior_nr(nr++);
  elements[8]->set_bot_edge_nr(nr);
  elements[9]->set_top_edge_nr(nr++);
  elements[9]->set_interior_nr(nr++);
  elements[9]->set_left_edge_nr(nr);
  elements[10]->set_right_edge_nr(nr++);
  elements[10]->set_interior_nr(nr++);
  elements[10]->set_left_edge_nr(nr);
  elements[11]->set_right_edge_nr(nr++);
  elements[11]->set_interior_nr(nr++);
  elements[11]->set_left_edge_nr(nr);
  elements[0]->set_right_edge_nr(nr);
}

void set_small_interface_nrs(int nr, Element** elements) {
  elements[11]->set_top_left_vertex_nr(nr);

  elements[0]->set_top_right_vertex_nr(nr);
  elements[1]->set_bot_right_vertex_nr(nr++);
  elements[1]->set_right_edge_nr(nr++);
  elements[1]->set_top_right_vertex_nr(nr);
  elements[2]->set_bot_right_vertex_nr(nr++);
  elements[2]->set_right_edge_nr(nr++);
  elements[2]->set_top_right_vertex_nr(nr);
  elements[3]->set_bot_right_vertex_nr(nr);
  elements[4]->set_bot_left_vertex_nr(nr++);
  elements[4]->set_bot_edge_nr(nr++);
  elements[4]->set_bot_right_vertex_nr(nr);
  elements[5]->set_bot_left_vertex_nr(nr++);
  elements[5]->set_bot_edge_nr(nr++);
  elements[5]->set_bot_right_vertex_nr(nr);
  elements[6]->set_bot_left_vertex_nr(nr);
  elements[7]->set_top_left_vertex_nr(nr++);
  elements[7]->set_left_edge_nr(nr++);
  elements[7]->set_bot_left_vertex_nr(nr);
  elements[8]->set_top_left_vertex_nr(nr++);
  elements[8]->set_left_edge_nr(nr++);
  elements[8]->set_bot_left_vertex_nr(nr);
  elements[9]->set_top_left_vertex_nr(nr);
  elements[10]->set_top_right_vertex_nr(nr++);
  elements[10]->set_top_edge_nr(nr++);
  elements[10]->set_top_left_vertex_nr(nr);
  elements[11]->set_top_right_vertex_nr(nr++);
  elements[11]->set_top_edge_nr(nr++);
}

void set_last_tier_interior_nrs(int nr, Element** elements) {
  elements[0]->set_bot_left_vertex_nr(nr++);
  elements[0]->set_left_edge_nr(nr++);
  elements[0]->set_top_left_vertex_nr(nr);
  elements[1]->set_bot_left_vertex_nr(nr++);
  elements[1]->set_left_edge_nr(nr++);
  elements[1]->set_top_left_vertex_nr(nr++);
  elements[1]->set_top_edge_nr(nr++);
  elements[1]->set_top_right_vertex_nr(nr);
  elements[2]->set_top_left_vertex_nr(nr++);
  elements[2]->set_top_edge_nr(nr++);
  elements[2]->set_top_right_vertex_nr(nr++);
  elements[2]->set_right_edge_nr(nr++);
  elements[2]->set_bot_right_vertex_nr(nr);
  elements[3]->set_top_right_vertex_nr(nr++);
  elements[3]->set_right_edge_nr(nr++);
  elements[3]->set_bot_right_vertex_nr(nr++);
  elements[3]->set_bot_edge_nr(nr++);
  elements[3]->set_bot_left_vertex_nr(nr);
  elements[0]->set_bot_right_vertex_nr(nr++);
  elements[0]->set_bot_edge_nr(nr++);

  elements[0]->set_interior_nr(nr++);
  elements[0]->set_top_edge_nr(nr);
  elements[1]->set_bot_edge_nr(nr++);
  elements[1]->set_interior_nr(nr++);
  elements[1]->set_right_edge_nr(nr);
  elements[2]->set_left_edge_nr(nr++);
  elements[2]->set_interior_nr(nr++);
  elements[2]->set_bot_edge_nr(nr);
  elements[3]->set_top_edge_nr(nr++);
  elements[3]->set_interior_nr(nr++);
  elements[3]->set_left_edge_nr(nr);
  elements[0]->set_right_edge_nr(nr++);

  elements[0]->set_top_right_vertex_nr(nr);
  elements[1]->set_bot_right_vertex_nr(nr);
  elements[2]->set_bot_left_vertex_nr(nr);
  elements[3]->set_top_left_vertex_nr(nr);
}

std::vector<EquationSystem*>*
MatrixGenerator::CreateMatrixAndRhs(TaskDescription& task_description) {

  tier_vector = new std::vector<EquationSystem*>();

  IDoubleArgFunction* f =
      new DoubleArgFunctionWrapper(task_description.function);
  double bot_left_x = task_description.x;
  double bot_left_y = task_description.y;
  int nr_of_tiers   = task_description.nrOfTiers;
  double size       = task_description.size;

  int nr                  = 0;
  int nr_of_tier_elements = 12;
  // outer interface + (interior+inner interface)*nr_of_tiers + last tier inner
  matrix_size = 32 + (24 + 16) * nr_of_tiers + 9;
  rhs         = new double[matrix_size]();
  matrix      = new double*[matrix_size];
  for (int i = 0; i < matrix_size; i++)
    matrix[i] = new double[matrix_size]();

  size /= 2.0;
  CreateTier(bot_left_x, bot_left_y, size, nr_of_tier_elements, nr, f, true);
  nr += 56;

  for (int i = 1; i < nr_of_tiers - 1; i++) {
    bot_left_x += size;
    bot_left_y += size;
    size /= 2.0;
    CreateTier(bot_left_x, bot_left_y, size, nr_of_tier_elements, nr, f, false);
    nr += 40;
  }
  bot_left_x += size;
  bot_left_y += size;
  size /= 2.0;
  // last tier with 16
  int nr_of_last_tier_elements = 16;
  Element** elements           = CreateElements(bot_left_x, bot_left_y, size,
                                      nr_of_last_tier_elements, false);
  for (int i = 0; i < 16 - nr_of_tier_elements; i++)
    element_list.push_back(elements[i]);
  set_big_interface_nrs(nr, false, elements);
  nr += 16;
  set_interior_nrs(nr, elements);
  nr += 24;
  set_small_interface_nrs(nr, elements);
  set_last_tier_interior_nrs(nr, elements + nr_of_tier_elements);
  tier_vector->push_back(
      new Tier(elements, f, matrix, rhs, nr_of_last_tier_elements));
  return tier_vector;
}

void MatrixGenerator::CreateTier(double x, double y, double element_size,
                                 int tier_size, int function_nr,
                                 IDoubleArgFunction* f, bool first_tier) {
  Element** elements =
      CreateElements(x, y, element_size, tier_size, first_tier);
  for (int i = 0; i < tier_size; i++)
    element_list.push_back(elements[i]);
  set_big_interface_nrs(function_nr, first_tier, elements);
  if (first_tier)
    function_nr += 32;
  else
    function_nr += 16;
  set_interior_nrs(function_nr, elements);
  function_nr += 24;
  set_small_interface_nrs(function_nr, elements);
  tier_vector->push_back(new Tier(elements, f, matrix, rhs, tier_size));
}

Element** MatrixGenerator::CreateElements(double x, double y,
                                          double element_size, int size,
                                          bool first_tier) {
  int i              = 0;
  Element** elements = new Element*[size];
  double coordinates[4];
  bool neighbours[4];
  neighbours[0]  = true;
  neighbours[1]  = true;
  neighbours[2]  = true;
  neighbours[3]  = true;
  coordinates[0] = x;
  coordinates[1] = x + element_size;
  coordinates[2] = y;
  coordinates[3] = y + element_size;

  if (!first_tier) {
    neighbours[LEFT] = false;
    neighbours[BOT]  = false;
  }
  elements[i++] = new Element(coordinates, neighbours, BOT_LEFT);
  coordinates[2] += element_size;
  coordinates[3] += element_size;
  if (!first_tier) {
    neighbours[BOT] = true;
  }
  elements[i++] = new Element(coordinates, neighbours, TOP_LEFT);
  coordinates[2] += element_size;
  coordinates[3] += element_size;
  elements[i++] = new Element(coordinates, neighbours, BOT_LEFT);
  coordinates[2] += element_size;
  coordinates[3] += element_size;
  if (!first_tier) {
    neighbours[TOP] = false;
  }
  elements[i++] = new Element(coordinates, neighbours, TOP_LEFT);
  coordinates[0] += element_size;
  coordinates[1] += element_size;
  if (!first_tier) {
    neighbours[LEFT] = true;
  }
  elements[i++] = new Element(coordinates, neighbours, TOP_RIGHT);
  coordinates[0] += element_size;
  coordinates[1] += element_size;
  elements[i++] = new Element(coordinates, neighbours, TOP_LEFT);
  coordinates[0] += element_size;
  coordinates[1] += element_size;
  if (!first_tier) {
    neighbours[RIGHT] = false;
  }
  elements[i++] = new Element(coordinates, neighbours, TOP_RIGHT);
  coordinates[2] -= element_size;
  coordinates[3] -= element_size;
  if (!first_tier) {
    neighbours[TOP] = true;
  }
  elements[i++] = new Element(coordinates, neighbours, BOT_RIGHT);
  coordinates[2] -= element_size;
  coordinates[3] -= element_size;
  elements[i++] = new Element(coordinates, neighbours, TOP_RIGHT);
  coordinates[2] -= element_size;
  coordinates[3] -= element_size;
  if (!first_tier) {
    neighbours[BOT] = false;
  }
  elements[i++] = new Element(coordinates, neighbours, BOT_RIGHT);
  coordinates[0] -= element_size;
  coordinates[1] -= element_size;
  if (!first_tier) {
    neighbours[RIGHT] = true;
  }
  elements[i++] = new Element(coordinates, neighbours, BOT_LEFT);
  coordinates[0] -= element_size;
  coordinates[1] -= element_size;
  elements[i++] = new Element(coordinates, neighbours, BOT_RIGHT);

  if (size == 16) {
    neighbours[0]  = true;
    neighbours[1]  = true;
    neighbours[2]  = true;
    neighbours[3]  = true;
    coordinates[0] = x + element_size;
    coordinates[1] = x + 2 * element_size;
    coordinates[2] = y + element_size;
    coordinates[3] = y + 2 * element_size;
    elements[i++]  = new Element(coordinates, neighbours, TOP_RIGHT);
    coordinates[2] += element_size;
    coordinates[3] += element_size;
    elements[i++] = new Element(coordinates, neighbours, BOT_RIGHT);
    coordinates[0] += element_size;
    coordinates[1] += element_size;
    elements[i++] = new Element(coordinates, neighbours, BOT_LEFT);
    coordinates[2] -= element_size;
    coordinates[3] -= element_size;
    elements[i] = new Element(coordinates, neighbours, TOP_LEFT);
  }

  return elements;
}

void MatrixGenerator::checkSolution(std::map<int, double>* solution_map,
                                    double (*function)(int dim, ...)) {
  IDoubleArgFunction* f            = new DoubleArgFunctionWrapper(*function);
  std::list<Element*>::iterator it = element_list.begin();
  bool solution_ok                 = true;
  while (it != element_list.end() && solution_ok) {
    Element* element = (*it);
    solution_ok      = element->checkSolution(solution_map, f);
    ++it;
  }
  if (solution_ok)
    printf("SOLUTION OK\n");
  else
    printf("WRONG SOLUTION\n");
}
