#include <iostream>
namespace galois {
constexpr uint32_t CHUNK_SIZE_64 = 64;

class UpdateRequestIndexer {
public:
  uint32_t shift;

  UpdateRequestIndexer(uint32_t _shift) : shift(_shift) {}
  template <typename R>
  unsigned int operator()(const R& req) const {
    unsigned int t = req.dist >> shift;
    return t;
  }
};

template <typename GNode, typename Dist>
struct UpdateRequest {
  GNode src;
  Dist dist;
  UpdateRequest(const GNode& N, Dist W) : src(N), dist(W) {}
  UpdateRequest() : src(), dist(0) {}

  friend bool operator<(const UpdateRequest& left, const UpdateRequest& right) {
    return left.dist == right.dist ? left.src < right.src
                                   : left.dist < right.dist;
  }
};

struct ReqPushWrap {
  template <typename C, typename GNode, typename Dist>
  void operator()(C& cont, const GNode& n, const Dist& dist) const {
    cont.push(UpdateRequest<GNode, Dist>(n, dist));
  }
};

} // namespace galois
