//#define USE_GAT
#ifdef USE_GAT
void graph_conv_layer::forward_propagation(const float_t* in_data,
                                           float_t* out_data) {
  galois::StatTimer conv_timer("GraphConvForward");
  conv_timer.start();
  size_t x = input_dims[0];
  size_t y = input_dims[1];
  size_t z = output_dims[1];

  // (1) dropout
  if (dropout_ && phase_ == net_phase::train) {
    math::dropout_cpu(x, y, scale_, dropout_rate_, in_data,
                      dropout_mask, in_temp);
  } else {
    math::copy_cpu(x * y, in_data, in_temp);
  }

  // (2) linear transformation
  math::sgemm_cpu(CblasNoTrans, CblasNoTrans, x, z, y, 1.0, in_temp,
                  &layer::W[0], 0.0, out_temp);

  auto &g = *graph_cpu;
  size_t n = g.size();
  size_t len = z;
  float_t* in = out_temp;
  float_t* out = out_data;
  
  galois::do_all(galois::iterate(size_t(0), n), [&](const auto src) {
    auto src_idx = src * len;
    auto deg_src = g.get_degree(src);

    // (3) concatenation, dot product, LeakyReLU
    int i = 0;
    vec_t scores(deg_src);
    //for (auto e : g.edges(src)) {
    for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
      auto dst = g.getEdgeDst(e);
      auto dst_idx = dst * len;
      vec_t concat_vec(2*z);
      math::concat(z, &in[src_idx], &in[dst_idx], &concat_vec[0]);
      // alpha: learnable weight vector
      scores[i++] = math::dot(2*z, &alpha[0], &concat_vec[0]);
    }

    // (4) softmax to normalize the attention scores on each vertex’s incoming edges
    vec_t normalized_scores(deg_src, 0);
    math::softmax(deg_src, &scores[0], &normalized_scores[0]); // normalize using softmax
    math::clear_cpu(len, &out[src_idx]);

    // (5) aggregation: scaled by the attention scores
    //for (auto e : g.edges(src)) {
    for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
      auto dst = g.getEdgeDst(e);
      auto dst_idx = dst * len;
      auto score = normalized_scores[dst];
      vec_t neighbor(len);
      math::scale(len, score, &in[dst_idx], &neighbor[0]);
      math::vadd_cpu(len, &out[src_idx], &neighbor[0], &out[src_idx]);
    }
  });
  
  // (6) ReLU
  if (act_) math::relu_cpu(x * z, out_data, out_data);
}
#endif
