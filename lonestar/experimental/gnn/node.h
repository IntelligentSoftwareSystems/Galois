#pragma once
#include <vector>
class node;
class layer;
class edge;

typedef std::shared_ptr<edge> edgeptr_t;

// node data structure
class node : public std::enable_shared_from_this<node> {
public:
  node(size_t in_size, size_t out_size) {
  } //: prev_(in_size), next_(out_size) {}
  virtual ~node() {}
  const edgeptr_t prev() const { return prev_; }
  // const std::vector<edgeptr_t> &prev() const { return prev_; }
  const edgeptr_t next() const { return next_; }
  // const std::vector<edgeptr_t> &next() const { return next_; }
  // std::vector<node *> prev_nodes() const;
  // std::vector<node *> next_nodes() const;

protected:
  node() = delete;
  friend void connect(layer* head, layer* tail, size_t head_index,
                      size_t tail_index);
  // mutable std::vector<edgeptr_t> prev_;
  // mutable std::vector<edgeptr_t> next_;
  mutable edgeptr_t prev_;
  mutable edgeptr_t next_;
};

// edges manage the input/output data and gradients between nodes
class edge {
public:
  edge(node* prev, size_t len)
      : ft_dim_(len), data_({vec_t(len)}), grad_({vec_t(len)}), prev_(prev) {}

  void merge_grads(vec_t* dst) {
    assert(!grad_.empty());
    const auto& grad_head = grad_[0];
    size_t sz             = grad_head.size();
    dst->resize(sz);
    float_t* pdst = &(*dst)[0];
    std::copy(grad_head.begin(), grad_head.end(), pdst);
    // @todo consider adding parallelism and vectorization
    for (size_t sample = 1; sample < grad_.size(); ++sample) {
      for (size_t i = 0; i < sz; i++)
        pdst[i] += grad_[sample][i];
      // vectorize::reduce<float_t>(&grad_[sample][0], sz, pdst);
    }
  }
  void clear_grads() {
    for (size_t sample = 0; sample < grad_.size(); ++sample) {
      auto& g = grad_[sample];
      std::fill(g.begin(), g.end(), 0.0); // TODO: need vectorize
      // vectorize::fill(&g[0], g.size(), float_t{0});
    }
  }

  tensor_t* get_data_ptr() { return &data_; }
  tensor_t& get_data() { return data_; }
  // const tensor_t *get_data() const { return &data_; }
  const tensor_t& get_data() const { return data_; }
  // tensor_t *get_gradient() { return &grad_; }
  tensor_t& get_gradient() { return grad_; }
  // const tensor_t *get_gradient() const { return &grad_; }
  const tensor_t& get_gradient() const { return grad_; }

  // const std::vector<node *> &next() const { return next_; }
  const node* next() const { return next_; }
  node* prev() { return prev_; }
  const node* prev() const { return prev_; }
  // const shape3d &shape() const { return shape_; }
  // vector_type vtype() const { return vtype_; }
  // void add_next_node(node *next) { next_.push_back(next); }
  void add_next_node(node* next) { next_ = next; }

private:
  // shape3d shape_;
  size_t ft_dim_;
  // vector_type vtype_;
  tensor_t data_;
  tensor_t grad_;
  node* prev_; // previous node, "producer" of this tensor
  node* next_; // next node, "consumer" of this tensor
  // std::vector<node *> next_;  // next nodes, "consumers" of this tensor
};
/*
inline std::vector<node *> node::prev_nodes() const {
    std::vector<node *> vecs;
    for (auto &e : prev_) {
        if (e && e->prev()) {
            vecs.insert(vecs.end(), e->prev());
        }
    }
    return vecs;
}

inline std::vector<node *> node::next_nodes() const {
    std::vector<node *> vecs;
    for (auto &e : next_) {
        if (e) {
            auto n = e->next();
            vecs.insert(vecs.end(), n.begin(), n.end());
        }
    }
    return vecs;
}
*/
