/**
 * Code taken/modified from below link.
 *
 * https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/optimizers/optimizer.h
 * Copyright (c) 2013, Taiga Nomi and the respective contributors
 * All rights reserved.
 * Reused under 3-BSD
 */
#pragma once

// TODO:
// - use classes, not structs (modern C++)
// - templatize this instead of using inheritance
// - put optimizers in their own namespace

#include <algorithm>
#include <unordered_map>
#include "deepgalois/types.h"

namespace deepgalois {

// base class of optimizer
// usesHessian : true if an optimizer uses hessian (2nd order derivative of loss
// function)
struct optimizer {
  optimizer()                 = default;
  optimizer(const optimizer&) = default;
  optimizer(optimizer&&)      = default;
  optimizer& operator=(const optimizer&) = default;
  optimizer& operator=(optimizer&&)              = default;
  virtual ~optimizer()                           = default;
  virtual void update(const vec_t& dW, vec_t& W) = 0;
#ifdef GALOIS_ENABLE_GPU
  virtual void update_gpu(const size_t n, const float_t* dW, float_t* W) = 0;
#endif
  virtual void reset() {} // override to implement pre-learning action
};

// helper class to hold N values for each weight
template <int N>
struct stateful_optimizer : public optimizer {
  void reset() override {
    for (auto& e : E_)
      e.clear();
  }

protected:
  template <int Index>
  vec_t& get(const vec_t& key) {
    static_assert(Index < N, "index out of range");
    if (E_[Index][&key].empty())
      E_[Index][&key].resize(key.size(), float_t(0));
    return E_[Index][&key];
  }
  std::unordered_map<const vec_t*, vec_t> E_[N];
#ifdef GALOIS_ENABLE_GPU
  template <int Index>
  float_t* get_gpu(const size_t n, const float_t* key);
  std::unordered_map<const float_t*, float_t*> dE_[N];
#endif
};

/**
 * adaptive gradient method
 *
 * J Duchi, E Hazan and Y Singer,
 * Adaptive subgradient methods for online learning and stochastic optimization
 * The Journal of Machine Learning Research, pages 2121-2159, 2011.
 **/
struct adagrad : public stateful_optimizer<1> {
  adagrad() : alpha(0.01), eps(float_t(1e-8)) {}
  void update(const vec_t& dW, vec_t& W);
#ifdef GALOIS_ENABLE_GPU
  void update_gpu(const size_t n, const float_t* dW, float_t* W);
#endif
  float_t alpha; // learning rate
private:
  float_t eps;
};

/**
 * RMSprop
 *
 * T Tieleman, and G E Hinton,
 * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
 **/
struct RMSprop : public stateful_optimizer<1> {
  RMSprop() : alpha(float_t(0.0001)), mu(float_t(0.99)), eps(float_t(1e-8)) {}
  void update(const vec_t& dW, vec_t& W);
#ifdef GALOIS_ENABLE_GPU
  void update_gpu(const size_t n, const float_t* dW, float_t* W);
#endif
  float_t alpha; // learning rate
  float_t mu;    // decay term
private:
  float_t eps; // constant value to avoid zero-division
};

// Adam: A Method for Stochastic Optimization
// http://arxiv.org/abs/1412.6980
struct adam : public stateful_optimizer<2> {
  adam()
      : alpha(float_t(0.01)), b1(float_t(0.9)), b2(float_t(0.999)),
        b1_t(float_t(0.9)), b2_t(float_t(0.999)), eps(float_t(1e-8)) {}
  void update(const vec_t& dW, vec_t& W);
#ifdef GALOIS_ENABLE_GPU
  void update_gpu(const size_t n, const float_t* dW, float_t* W);
#endif

  float_t alpha; // learning rate
  float_t b1;    // decay term
  float_t b2;    // decay term
  float_t b1_t;  // decay term power t
  float_t b2_t;  // decay term power t

private:
  float_t eps; // constant value to avoid zero-division
};

/**
 * @brief [a new optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 2)
 *               http://arxiv.org/abs/1412.6980]
 *
 */
struct adamax : public stateful_optimizer<2> {
  adamax()
      : alpha(float_t(0.002)), b1(float_t(0.9)), b2(float_t(0.999)), b1_t(b1),
        eps(float_t(1e-8)) {}
  void update(const vec_t& dW, vec_t& W);
#ifdef GALOIS_ENABLE_GPU
  void update_gpu(const size_t n, const float_t* dW, float_t* W);
#endif

  float_t alpha; // learning rate
  float_t b1;    // decay term
  float_t b2;    // decay term
  float_t b1_t;  // decay term power t

private:
  float_t eps; // constant value to avoid zero-division
};

// SGD without momentum
// slightly faster than tiny_dnn::momentum
struct gradient_descent : public optimizer {
  gradient_descent() : alpha(float_t(0.01)), lambda(float_t(0)) {}
  void update(const vec_t& dW, vec_t& W);
#ifdef GALOIS_ENABLE_GPU
  void update_gpu(const size_t n, const float_t* dW, float_t* W);
#endif
  float_t alpha;  // learning rate
  float_t lambda; // weight decay
};

/**
 * SGD with momentum
 *
 * B T Polyak,
 * Some methods of speeding up the convergence of iteration methods
 * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
 **/
struct momentum : public stateful_optimizer<1> {
public:
  momentum() : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}
  void update(const vec_t& dW, vec_t& W);
#ifdef GALOIS_ENABLE_GPU
  void update_gpu(const size_t n, const float_t* dW, float_t* W);
#endif

  float_t alpha;  // learning rate
  float_t lambda; // weight decay
  float_t mu;     // momentum
};

/**
 * SGD with Nesterov momentum
 *
 * Y Nesterov,
 * A method for unconstrained convex minimization problem with the rate of
 * convergence o(1/k2), Doklady ANSSSR, vol.269, pp.543-547, 1983.
 **/
struct nesterov_momentum : public stateful_optimizer<1> {
public:
  nesterov_momentum()
      : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}
  void update(const vec_t& dW, vec_t& W);
#ifdef GALOIS_ENABLE_GPU
  void update_gpu(const size_t n, const float_t* dW, float_t* W);
#endif

  float_t alpha;  // learning rate
  float_t lambda; // weight decay
  float_t mu;     // momentum
};

} // namespace deepgalois
