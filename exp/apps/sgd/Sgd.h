#ifndef SGD_H
#define SGD_H

#include <cassert>
#include <string>

typedef double LatentValue;
static const int LATENT_VECTOR_SIZE = 100; // Purdue, CSGD: 100; Intel: 20

// like std::inner_product but rewritten here to check vectorization
template<typename T>
T innerProduct(
    T* __restrict__ first1,
    T* __restrict__ last1,
    T* __restrict__ first2,
    T init) {
  assert(first1 + LATENT_VECTOR_SIZE == last1);
  for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
    init += first1[i] * first2[i];
  return init;
}

template<typename T>
T predictionError(
    T* __restrict__ movieLatent,
    T* __restrict__ userLatent,
    double actual)
{
  T v = actual;
  return innerProduct(movieLatent, movieLatent + LATENT_VECTOR_SIZE, userLatent, -v);
}

// Objective: squared loss with weighted-square-norm regularization
template<typename T>
T doGradientUpdate(
    T* __restrict__ movieLatent,
    T* __restrict__ userLatent,
    double lambda,
    double edgeRating,
    double stepSize) 
{
  T l = lambda;
  T step = stepSize;
  T rating = edgeRating;
  T error = innerProduct(movieLatent, movieLatent + LATENT_VECTOR_SIZE, userLatent, -rating);

  // Take gradient step
  for (int i = 0; i < LATENT_VECTOR_SIZE; i++) {
      T prevMovie = movieLatent[i];
      T prevUser = userLatent[i];
      movieLatent[i] -= step * (error * prevUser  + l * prevMovie);
      userLatent[i]  -= step * (error * prevMovie + l * prevUser);
  }

  return error;
}

struct StepFunction {
  virtual LatentValue stepSize(int round) const = 0;
  virtual std::string name() const = 0;
  virtual bool isBold() const { return false; }
};

StepFunction* newStepFunction();

template<typename Graph>
size_t initializeGraphData(Graph& g);

#endif
