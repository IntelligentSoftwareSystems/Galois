#include "MatrixGenerator.hxx"

using namespace D3;
std::vector<EquationSystem*>*
MatrixGenerator::CreateMatrixAndRhs(TaskDescription& task_description) {

  TripleArgFunctionWrapper* f =
      new TripleArgFunctionWrapper(task_description.function);
  int nr_of_tiers        = task_description.nrOfTiers;
  double bot_left_near_x = task_description.x;
  double bot_left_near_y = task_description.y;
  double bot_left_near_z = task_description.z;
  double size            = task_description.size;

  int nr             = 0;
  bool neigbours[18] = {true};
  Element* element =
      new Element(bot_left_near_x, bot_left_near_y + size / 2.0,
                  bot_left_near_z, size / 2.0, neigbours, BOT_LEFT_FAR);

  Element** elements = element->CreateFirstTier(nr);

  for (int i = 0; i < 7; i++)
    element_list.push_back(elements[i]);

  delete[] elements;
  // nr of degrees of freedom in first tier which are not in 2
  nr += 98;

  double x             = bot_left_near_x;
  double y             = bot_left_near_y + size / 2.0;
  double z             = bot_left_near_z;
  double s             = size / 2.0;
  neigbours[LEFT2]     = false;
  neigbours[BOT_LEFT]  = false;
  neigbours[LEFT_NEAR] = false;
  neigbours[LEFT_FAR]  = false;
  neigbours[TOP_LEFT]  = false;
  neigbours[FAR]       = false;
  neigbours[TOP_FAR]   = false;
  neigbours[BOT_FAR]   = false;
  neigbours[RIGHT_FAR] = false;
  for (int i = 1; i < nr_of_tiers; i++) {

    x        = x + s;
    y        = y - s / 2.0;
    s        = s / 2.0;
    element  = new Element(x, y, z, s, neigbours, BOT_LEFT_FAR);
    elements = element->CreateAnotherTier(nr);

    for (int i = 0; i < 7; i++)
      element_list.push_back(elements[i]);
    // nr of degrees of freedom in i tier which are not in i + 1
    delete[] elements;
    nr += 56;
  }

  x = x + s;
  y = y - s;
  for (int i = 0; i < 18; i++)
    neigbours[i] = true;
  element = new Element(x, y, z, s, neigbours, BOT_RIGHT_NEAR);
  element->CreateLastTier(nr);
  element_list.push_back(element);

  matrix_size = 98 + 56 * (nr_of_tiers - 1) + 27;
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
      tier = new Tier(*it, *(++it), *(++it), *(++it), *(++it), *(++it), *(++it),
                      *(++it), f, matrix, rhs);
    } else {
      tier = new Tier(*it, *(++it), *(++it), *(++it), *(++it), *(++it), *(++it),
                      NULL, f, matrix, rhs);
      ++it;
    }
    tier_vector->push_back(tier);
  }

  return tier_vector;
}

void MatrixGenerator::checkSolution(std::map<int, double>* solution_map,
                                    double (*function)(int dim, ...)) {
  srand(time(NULL));
  TripleArgFunctionWrapper* f      = new TripleArgFunctionWrapper(function);
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
