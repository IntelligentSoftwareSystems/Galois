// #define USE_GAT
#ifdef USE_GAT
// `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`
// NOTE: GAT paper uses "first concatenation then linear projection"
//  to compute attention scores, while ours is "first projection then
//  addition", the two approaches are mathematically equivalent:
//  We decompose the weight vector a mentioned in the paper into
//  [a_l || a_r], then  a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
//  Our implementation is much efficient because we do not need to
//  save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
//  addition could be optimized with DGL's built-in function u_add_v,
//  which further speeds up computation and saves memory footprint.

void graph_conv_layer::aggregate(size_t len, Graph& g, const float_t* in,
                                 float_t* out) {
  size_t n = g.size();
  galois::do_all(galois::iterate(size_t(0), n), [&](const auto src) {
    auto src_idx = src * len;
    auto deg_src = g.get_degree(src);

    // concatenation, dot product, LeakyReLU
    // int i = 0;
    // vec_t scores(deg_src);
    auto begin = g.edge_begin(src);
    auto end   = g.edge_end(src);
    // alpha: learnable weight vector (shared by all vertices)
    float_t src_score = math::dot(len, &alpha_l[0], &in[src_idx]);
    for (auto e = begin; e != end; e++) {
      auto dst     = g.getEdgeDst(e);
      auto dst_idx = dst * len;
      // vec_t concat_vec(2*len);
      // math::concat(len, &in[src_idx], &in[dst_idx], &concat_vec[0]);
      // float_t score = math::dot(2*len, &alpha[0], &concat_vec[0]);
      float_t dst_score = math::dot(len, &alpha_r[0], &in[dst_idx]);
      temp_scores[e]    = src_score + dst_score;
      math::leaky_relu(epsilon, temp_scores[e], scores[e]);
    }

    // softmax to normalize the attention scores on each vertex’s incoming edges
    // vec_t normalized_scores(deg_src, 0);
    // math::softmax(deg_src, &scores[0], &normalized_scores[0]);
    math::softmax(deg_src, &scores[begin], &norm_scores[begin]);

    // aggregation: scaled by the attention scores
    math::clear_cpu(len, &out[src_idx]);
    for (auto e = begin; e != end; e++) {
      auto dst     = g.getEdgeDst(e);
      auto dst_idx = dst * len;
      auto score   = norm_scores[e];
      vec_t neighbor(len);
      math::scale(len, score, &in[dst_idx], &neighbor[0]);
      math::vadd_cpu(len, &out[src_idx], &neighbor[0], &out[src_idx]);
    }
  });
}

void graph_conv_layer::d_compute_scores(size_t len, Graph& g,
                                        const float_t* in_data,
                                        const float_t* out_data,
                                        const float_t* in_grad) {
  size_t n = g.size();

  // compute gradients for the learnable vector `alpha`
  // vec_t temp_grad(n*n);
  // math::sgemm_cpu(CblasTrans, CblasNoTrans, n, len, n, 1.0, out_data,
  //                in_grad, 0.0, temp_grad);
  galois::do_all(galois::iterate(size_t(0), n), [&](const auto src) {
    auto begin   = g.edge_begin(src);
    auto end     = g.edge_end(src);
    auto deg_src = g.get_degree(src);
    math::d_softmax(deg_src, &scores[begin], &norm_scores[begin],
                    &scores_grad[begin], &norm_scores_grad[begin]);
    for (auto e = begin; e != end; e++) {
      auto dst = g.getEdgeDst(e);
      // use norm_scores_grad as temp_scores_grad since its data is useless
      // already
      math::d_leaky_relu(epsilon, &scores_grad[e], &temp_scores[e],
                         &norm_scores_grad[e]);
      math::scale(len, norm_scores_grad[e], &in_data[src_idx], &alpha_lgrad[0]);
      math::scale(len, norm_scores_grad[e], &in_data[dst_idx], &alpha_rgrad[0]);
    }
  });
}

void graph_conv_layer::d_aggregate(size_t len, Graph& g, const float_t* in_grad,
                                   float_t* out_grad) {
  size_t n = g.size();

  // aggregation: the derivative is transposed;
  // the graph is undirected (structurally symmetric),
  // but values are not the same for the symmetric positions
  galois::do_all(galois::iterate(size_t(0), n), [&](const auto src) {
    auto src_idx   = src * len;
    auto src_begin = g.edge_begin(src);
    for (auto e = src_begin; e != g.edge_end(src); e++) {
      auto dst       = g.getEdgeDst(e);
      auto dst_idx   = dst * len;
      auto dst_begin = g.edge_begin(dst);
      auto score     = norm_scores[dst_begin + e - src_begin]; // transposed
      vec_t neighbor(len);
      math::scale(len, score, &in_grad[dst_idx], &neighbor[0]);
      math::vadd_cpu(len, &out_grad[src_idx], &neighbor[0], &out_grad[src_idx]);
    }
  });
}

void graph_conv_layer::forward_propagation(const float_t* in_data,
                                           float_t* out_data) {
  galois::StatTimer conv_timer("GraphConvForward");
  conv_timer.start();
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  size_t z = output_dims[1];

  // dropout
  if (dropout_ && phase_ == net_phase::train) {
    math::dropout_cpu(x, y, scale_, dropout_rate_, in_data, dropout_mask,
                      in_temp);
  } else {
    math::copy_cpu(x * y, in_data, in_temp);
  }

  // linear transformation
  math::sgemm_cpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_temp,
                  &layer::W[0], 0.0, out_temp);

  // aggregation
  aggregate(z, *graph_cpu, out_temp, out_data);

  // ReLU
  if (act_)
    math::relu_cpu(x * z, out_data, out_data);
}

void graph_conv_layer::back_propagation(const float_t* in_data,
                                        const float_t* out_data,
                                        float_t* out_grad, float_t* in_grad) {
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  size_t z = output_dims[1];
  if (act_)
    math::d_relu_cpu(x * z, out_grad, out_data, out_grad);

  // compute gradients for alpha (alpha is a learnable vector)
  d_compute_scores(z, *graph_cpu, in_temp, out_temp, out_grad);
  // compute gradients for feature vectors
  d_aggregate(z, *graph_cpu, out_grad, out_temp);
  if (level_ != 0) {
    math::sgemm_cpu(CblasNoTrans, CblasTrans, x, y, z, 1.0, out_temp, &W[0],
                    0.0, in_grad); // x*z; z*y -> x*y
    math::sgemm_cpu(CblasTrans, CblasNoTrans, y, z, x, 1.0, in_data, out_temp,
                    0.0, &layer::weight_grad[0]); // y*x; x*z; y*z
  }
  if (level_ != 0 && dropout_)
    math::d_dropout_cpu(x, y, scale_, in_grad, dropout_mask, in_grad);
}

#endif
