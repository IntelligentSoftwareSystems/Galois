#include "MatrixGenerator.hxx"

using namespace D2;
std::vector<EquationSystem*>*
MatrixGenerator::CreateMatrixAndRhs(TaskDescription& task_description) {
  double* coordinates = new double[4];
  bool* neighbours    = new bool[4];

  IDoubleArgFunction* f =
      new DoubleArgFunctionWrapper(task_description.function);
  double bot_left_x = task_description.x;
  double bot_left_y = task_description.y;
  int nr_of_tiers   = task_description.nrOfTiers;
  double size       = task_description.size;

  int nr            = 0;
  neighbours[LEFT]  = true;
  neighbours[TOP]   = true;
  neighbours[RIGHT] = true;
  neighbours[BOT]   = true;
  coordinates[0]    = bot_left_x;
  coordinates[1]    = bot_left_x + size / 2.0;
  coordinates[2]    = bot_left_y + size / 2.0;
  coordinates[3]    = bot_left_y + size;
  Element* element  = new Element(coordinates, neighbours, TOP_LEFT);

  Element** elements = element->CreateFirstTier(nr);

  element_list.push_back(element);
  element_list.push_back(elements[0]);
  element_list.push_back(elements[1]);
  nr += 16;

  double x         = bot_left_x;
  double y         = bot_left_y + size / 2.0;
  double s         = size / 2.0;
  neighbours[LEFT] = false;
  neighbours[TOP]  = false;
  for (int i = 1; i < nr_of_tiers; i++) {

    x              = x + s;
    y              = y - s / 2.0;
    s              = s / 2.0;
    coordinates[0] = x;
    coordinates[1] = x + s;
    coordinates[2] = y;
    coordinates[3] = y + s;
    element        = new Element(coordinates, neighbours, TOP_LEFT);
    elements       = element->CreateAnotherTier(nr);

    element_list.push_back(element);
    element_list.push_back(elements[0]);
    element_list.push_back(elements[1]);
    nr += 12;
  }

  x                 = x + s;
  y                 = y - s;
  coordinates[0]    = x;
  coordinates[1]    = x + s;
  coordinates[2]    = y;
  coordinates[3]    = y + s;
  neighbours[LEFT]  = true;
  neighbours[TOP]   = true;
  neighbours[RIGHT] = true;
  neighbours[BOT]   = true;
  element           = new Element(coordinates, neighbours, BOT_RIGHT);
  element->CreateLastTier(nr);
  element_list.push_back(element);

  matrix_size = 9 + nr_of_tiers * 12 + 4;
  rhs         = new double[matrix_size]();
  matrix      = new double*[matrix_size];
  for (int i = 0; i < matrix_size; i++)
    matrix[i] = new double[matrix_size]();

  tier_vector                      = new std::vector<EquationSystem*>();
  std::list<Element*>::iterator it = element_list.begin();
  it                               = element_list.begin();

  for (int i = 0; i < nr_of_tiers; i++) {
    Tier* tier;
    if (i == nr_of_tiers - 1) {
      tier = new Tier(*it, *(++it), *(++it), *(++it), f, matrix, rhs);
    } else {
      tier = new Tier(*it, *(++it), *(++it), NULL, f, matrix, rhs);
      ++it;
    }
    tier_vector->push_back(tier);
  }
  // xyz
  delete[] elements;
  delete[] coordinates;
  delete[] neighbours;
  return tier_vector;
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
