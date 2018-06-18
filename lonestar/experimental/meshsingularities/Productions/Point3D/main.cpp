#include <stdio.h>
#include "MatrixGenerator.hxx"
#include <map>
#include <time.h>
#include <map>
#include "../EquationSystem.h"

using namespace D3;

double test_function(int dim, ...) {
  double* data  = new double[dim];
  double result = 0;
  va_list args;

  va_start(args, dim);
  for (int i = 0; i < dim; ++i) {
    data[i] = va_arg(args, double);
  }
  va_end(args);

  if (dim == 3) {
    double x = data[0];
    double y = data[1];
    double z = data[2];
    result   = 3 * x * x + 2 * y * y + z * z + x * y * z + x * x * z + y * y +
             2 * x * x * y * y * z * z;
  } else {
    result = -1;
  }

  delete[] data;
  return result;
}

int main(int argc, char** argv) {

  int nrOfTiers = 4;
  int i         = 0;

  GenericMatrixGenerator* matrixGenerator = new MatrixGenerator();

  TaskDescription taskDescription;
  taskDescription.dimensions = 3;
  taskDescription.nrOfTiers  = 4;
  taskDescription.size       = 1;
  taskDescription.function   = test_function;
  taskDescription.x          = -1;
  taskDescription.y          = -1;
  taskDescription.z          = -1;
  std::vector<EquationSystem*>* tiers =
      matrixGenerator->CreateMatrixAndRhs(taskDescription);
  EquationSystem* globalSystem = new EquationSystem(
      matrixGenerator->GetMatrix(), matrixGenerator->GetRhs(),
      matrixGenerator->GetMatrixSize());

  globalSystem->eliminate(matrixGenerator->GetMatrixSize());
  globalSystem->backwardSubstitute(matrixGenerator->GetMatrixSize() - 1);
  std::map<int, double>* result_map = new std::map<int, double>();
  for (int i = 0; i < matrixGenerator->GetMatrixSize(); i++) {
    (*result_map)[i] = globalSystem->rhs[i];
  }

  matrixGenerator->checkSolution(result_map, test_function);

  delete matrixGenerator;
  delete result_map;
  return 0;
}
