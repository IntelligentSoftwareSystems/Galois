#ifndef __MATRIXGENERATOR_2DEDGE_H_INCLUDED
#define __MATRIXGENERATOR_2DEDGE_H_INCLUDED

#include "../Point2D/DoubleArgFunction.hxx"
#include "../Point2D/Element.hxx"
#include "../Point2D/EPosition.hxx"
#include "Tier.hxx"
#include "../MatrixGeneration/GenericMatrixGenerator.hxx"
#include <vector>
#include <map>
#include <cmath>
namespace D2Edge {
class MatrixGenerator : public GenericMatrixGenerator {

private:
  int elements_size;
  Element** elements;
  int nr_of_elements;
  void CreateTiers(int to_create, int element_id, double size,
                   double* coordinates, IDoubleArgFunction* f, bool first_tier);

  // mumps
  bool mumps_arrays_created;
  int* in;
  int* jn;
  double* a;
  int n;
  int nz;

public:
  virtual std::vector<EquationSystem*>*
  CreateMatrixAndRhs(TaskDescription& task_description);
  virtual void checkSolution(std::map<int, double>* solution_map,
                             double (*f)(int dim, ...));
  virtual bool GetMumpsArrays(int*& in, int*& jn, double*& a, double*& rhs,
                              int& n, int& nz);
  virtual std::vector<int>* GetProductionParameters(int polynomial_degree);

  virtual ~MatrixGenerator() {
    for (int i = 0; i < elements_size; i++)
      delete elements[i];
    delete[] elements;
  }
};
} // namespace D2Edge
#endif
