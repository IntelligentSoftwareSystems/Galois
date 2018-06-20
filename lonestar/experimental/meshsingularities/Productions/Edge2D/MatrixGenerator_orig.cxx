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
#include <sys/time.h>

using namespace D2Edge;

void set_numbers(Element** elements, int row_nr, int last_row_nr,
                 int start_nr) {
  int first_element_in_row_nr = pow(2, row_nr + 1) - 2;
  int nr_of_elements_in_row   = pow(2, row_nr + 1);
  if (row_nr == last_row_nr)
    nr_of_elements_in_row /= 2;

  int next_start_nr;

  for (int i = 0; i < nr_of_elements_in_row; i += 2) {
    Element* left_element    = elements[i + first_element_in_row_nr];
    Element* right_element   = elements[i + first_element_in_row_nr + 1];
    bool top_edge_constraint = (row_nr != 0 && row_nr != last_row_nr);
    left_element->set_top_left_vertex_nr(start_nr++);
    left_element->set_top_edge_nr(start_nr++);
    left_element->set_top_right_vertex_nr(start_nr);
    if (top_edge_constraint)
      start_nr -= 2;

    right_element->set_top_left_vertex_nr(start_nr++);
    right_element->set_top_edge_nr(start_nr++);
    right_element->set_top_right_vertex_nr(start_nr);
    if (i + 2 == nr_of_elements_in_row)
      start_nr++;
  }

  for (int i = 0; i < nr_of_elements_in_row; i += 2) {
    Element* left_element  = elements[i + first_element_in_row_nr];
    Element* right_element = elements[i + first_element_in_row_nr + 1];
    left_element->set_left_edge_nr(start_nr++);
    left_element->set_interior_nr(start_nr++);
    left_element->set_right_edge_nr(start_nr);

    right_element->set_left_edge_nr(start_nr++);
    right_element->set_interior_nr(start_nr++);
    right_element->set_right_edge_nr(start_nr);
    if (i + 2 == nr_of_elements_in_row)
      start_nr++;
  }

  next_start_nr = start_nr;

  for (int i = 0; i < nr_of_elements_in_row; i += 2) {
    Element* left_element  = elements[i + first_element_in_row_nr];
    Element* right_element = elements[i + first_element_in_row_nr + 1];
    left_element->set_bot_left_vertex_nr(start_nr++);
    left_element->set_bot_edge_nr(start_nr++);
    left_element->set_bot_right_vertex_nr(start_nr);

    right_element->set_bot_left_vertex_nr(start_nr++);
    right_element->set_bot_edge_nr(start_nr++);
    right_element->set_bot_right_vertex_nr(start_nr);
    if (i + 2 == nr_of_elements_in_row)
      start_nr++;
  }
  if (row_nr + 1 <= last_row_nr)
    set_numbers(elements, row_nr + 1, last_row_nr, next_start_nr);
}

void MatrixGenerator::CreateTiers(int to_create, int element_id, double size,
                                  double* coordinates, IDoubleArgFunction* f,
                                  bool first_tier) {

  bool neighbours[4];
  neighbours[0] = true;
  neighbours[1] = true;
  neighbours[2] = true;
  neighbours[3] = true;

  if (to_create == 1) {

    coordinates[2] -= size;
    coordinates[3] -= size;
    Element* element;
    // position doesn't matter
    if (element_id % 2)
      element = new Element(coordinates, neighbours, BOT_RIGHT);
    else
      element = new Element(coordinates, neighbours, BOT_LEFT);

    int parent_id = element_id;
    if (element_id % 4)
      parent_id -= 2;
    parent_id /= 2;

    int row_nr = 0;
    while ((pow(2, row_nr + 1) - 2) < parent_id) {

      row_nr++;
    }
    if (((pow(2, row_nr + 1) - 2) == parent_id) && (element_id % 4))
      row_nr++;

    if (element_id % 4)
      element_id = parent_id + pow(2, row_nr);
    else
      element_id = parent_id + pow(2, row_nr) - 1;

    elements[element_id] = element;
    tier_vector->push_back(new Tier(element, f, matrix, rhs));
    return;
  }

  double xl;
  double xr;
  double yl;
  double yr;
  size /= 2.0;

  if (first_tier) {

    coordinates[1] = coordinates[0] + size;
    coordinates[2] += size;
    coordinates[3] = coordinates[2] + size;

  } else {
    neighbours[TOP] = false;
    coordinates[1] -= size;
    coordinates[2] -= size;
    coordinates[3] -= 2 * size;
  }

  xl                    = coordinates[0];
  xr                    = coordinates[1];
  yl                    = coordinates[2];
  yr                    = coordinates[3];
  Element* left_element = new Element(coordinates, neighbours, TOP_LEFT);
  CreateTiers((to_create - 2) / 2, (element_id + 1) * 2, size, coordinates, f,
              false);
  coordinates[0] = xl;
  coordinates[1] = xr;
  coordinates[2] = yl;
  coordinates[3] = yr;
  coordinates[0] += size;
  coordinates[1] += size;
  Element* right_element = new Element(coordinates, neighbours, TOP_RIGHT);
  CreateTiers((to_create - 2) / 2, (element_id + 1) * 2 + 2, size, coordinates,
              f, false);
  elements[element_id]     = left_element;
  elements[element_id + 1] = right_element;
  tier_vector->push_back(new Tier(left_element, f, matrix, rhs));
  tier_vector->push_back(new Tier(right_element, f, matrix, rhs));
}

// if nr_of_tiers = 1 then nr_of_elements = 10 !

std::vector<EquationSystem*>*
MatrixGenerator::CreateMatrixAndRhs(TaskDescription& task_description) {
  tier_vector = new std::vector<EquationSystem*>();

  IDoubleArgFunction* f =
      new DoubleArgFunctionWrapper(task_description.function);
  double bot_left_x = task_description.x;
  double bot_left_y = task_description.y;
  int nr_of_tiers   = task_description.nrOfTiers;
  double size       = task_description.size;

  nr_of_elements = 6 * pow(2, nr_of_tiers) - 2;
  elements       = new Element*[nr_of_elements];

  double coordinates[4];
  coordinates[0] = bot_left_x;
  coordinates[1] = 0;
  coordinates[2] = bot_left_y;
  coordinates[3] = 0;

  matrix_size = 3 * pow(2, nr_of_tiers + 3) + 2 * nr_of_tiers + 1;

  rhs = new double[matrix_size]();
  // matrix is too large to be creted in this type of singularity, therefore we
  // will create mumps arrays directly
  matrix = NULL;
  /*matrix = new double*[matrix_size];
  for(int i = 0; i<matrix_size; i++)
      matrix[i] = new double[matrix_size]();
      */

  CreateTiers(nr_of_elements, 0, size, coordinates, f, true);

  set_numbers(elements, 0, 1 + nr_of_tiers, 0);
  std::vector<EquationSystem*>::iterator it_e = tier_vector->begin();
  for (; it_e != tier_vector->end(); ++it_e) {
    ((Tier*)(*it_e))->InitTier();
  }

  // zmiana
  it_e = tier_vector->begin();
  for (; it_e != tier_vector->end(); ++it_e) {
    // printf("--------------------------\n");
    //(*it_e)->print();
  }
  // zmiana

  return tier_vector;
}

bool MatrixGenerator::GetMumpsArrays(int*& _in, int*& _jn, double*& _a,
                                     double*& _rhs, int& _n, int& _nz) {
  if (!mumps_arrays_created) {
    std::vector<EquationSystem*>::iterator it_t = tier_vector->begin();
    std::map<std::pair<int, int>, double>* map =
        new std::map<std::pair<int, int>, double>();
    for (; it_t != tier_vector->end(); ++it_t) {
      ((Tier*)(*it_t))->FillNumberPairs(map, rhs);
    }
    long mumps_array_length = map->size();

    in = new int[mumps_array_length]();
    jn = new int[mumps_array_length]();
    a  = new double[mumps_array_length]();

    std::map<std::pair<int, int>, double>::iterator it_m = map->begin();
    int i                                                = 0;
    for (; it_m != map->end(); ++it_m) {
      in[i]  = it_m->first.first;
      jn[i]  = it_m->first.second;
      a[i++] = it_m->second;
    }

    delete map;

    n = matrix_size;
    //???? is that ok? what should be nz value if the input is triangle of
    //diagonal matrix? non zeros in global matrix or just length of mumps
    //arrays? nz = 2*mumps_array_length - n;
    nz = mumps_array_length;
    // nz = 18;
  }

  // zmiana
  // for(int i = 0; i<nz; i++)
  // printf("%d %d %lf\n",in[i],jn[i],a[i]);

  timeval start_time;
  timeval end_time;
  gettimeofday(&start_time, NULL);
  for (int i = 0; i < nz; i++) {
  }
  gettimeofday(&end_time, NULL);
  printf("Tiers time %f [s] \n",
         ((end_time.tv_sec - start_time.tv_sec) * 1000000 +
          (end_time.tv_usec - start_time.tv_usec)) /
             1000000.0);
  printf("%d\n", nz);
  // zmiana

  mumps_arrays_created = true;
  _in                  = in;
  _jn                  = jn;
  _a                   = a;
  _rhs                 = rhs;
  _n                   = n;
  _nz                  = nz;
  return true;
}

std::vector<int>*
MatrixGenerator::GetProductionParameters(int polynomial_degree) {
  std::vector<int>* param = new std::vector<int>(1);
  (*param)[0]             = matrix_size;
  return param;
}

void MatrixGenerator::checkSolution(std::map<int, double>* solution_map,
                                    double (*function)(int dim, ...)) {

  srand(time(NULL));
  IDoubleArgFunction* f = new DoubleArgFunctionWrapper(*function);
  int i                 = 0;

  bool solution_ok = true;
  while (i < nr_of_elements && solution_ok) {
    Element* element = elements[i];
    solution_ok      = element->checkSolution(solution_map, f);
    ++i;
  }
  if (solution_ok)
    printf("SOLUTION OK\n");
  else
    printf("WRONG SOLUTION\n");
}
