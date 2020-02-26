#include "optimizer.h"
#include "galois/Galois.h"

void adagrad::update(const vec_t& dW, vec_t& W, bool parallelize) {
  vec_t& g = get<0>(W);
  if (parallelize) {
    galois::do_all(galois::iterate((size_t)0, W.size()),
      [&](const auto& i) {
        g[i] += dW[i] * dW[i];
        W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
      }, galois::loopname("adagrad_update"));
  } else {
    for (size_t i = 0; i < W.size(); i++) {
      g[i] += dW[i] * dW[i];
      W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
    }
  }
}

void RMSprop::update(const vec_t& dW, vec_t& W, bool parallelize) {
  vec_t& g = get<0>(W);
  galois::do_all(galois::iterate((size_t)0, W.size()),
    [&](const auto& i) {
      g[i] = mu * g[i] + (1 - mu) * dW[i] * dW[i];
      W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);
    }, galois::loopname("rms_update"));
}

void adam::update(const vec_t& dW, vec_t& W, bool parallelize) {
  vec_t& mt = get<0>(W);
  vec_t& vt = get<1>(W);
  galois::do_all(galois::iterate((size_t)0, W.size()),
    [&](const auto& i) {
      mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
      vt[i] = b2 * vt[i] + (float_t(1) - b2) * dW[i] * dW[i];
      // L2 norm based update rule
      W[i] -= alpha * (mt[i] / (float_t(1) - b1_t)) /
              std::sqrt((vt[i] / (float_t(1) - b2_t)) + eps);
    }, galois::chunk_size<256>(), galois::steal(),
    galois::loopname("adam_update"));
  b1_t *= b1;
  b2_t *= b2;
}

void adamax::update(const vec_t& dW, vec_t& W, bool parallelize) {
  vec_t& mt = get<0>(W);
  vec_t& ut = get<1>(W);
  galois::do_all(galois::iterate((size_t)0, W.size()),
    [&](const auto& i) {
      mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
      ut[i] = std::max(b2 * ut[i], std::abs(dW[i]));
      // Lp norm based update rule
      W[i] -= (alpha / (1.0 - b1_t)) * (mt[i] / (ut[i] + eps));
    }, galois::loopname("adamax_update"));
  b1_t *= b1;
}

void gradient_descent::update(const vec_t& dW, vec_t& W, bool parallelize) {
  galois::do_all(galois::iterate((size_t)0, W.size()),
      [&](const auto& i) { W[i] = W[i] - alpha * (dW[i] + lambda * W[i]); },
    galois::loopname("gradient_descent_update"));
}

void momentum::update(const vec_t& dW, vec_t& W, bool parallelize) {
  vec_t& dWprev = get<0>(W);
  galois::do_all(galois::iterate((size_t)0, W.size()),
    [&](const auto& i) {
      float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
      W[i] += V;
      dWprev[i] = V;
    }, galois::loopname("momentum_update"));
}

void nesterov_momentum::update(const vec_t& dW, vec_t& W, bool parallelize) {
  vec_t& dWprev = get<0>(W);
  galois::do_all(galois::iterate((size_t)0, W.size()),
    [&](const auto& i) {
      float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
      W[i] += (-mu) * dWprev[i] + (1 + mu) * V;
      dWprev[i] = V;
    }, galois::loopname("nesterov_momentum_update"));
}
