static const unsigned int DIST_INFINITY =
  std::numeric_limits<unsigned int>::max() - 1;

template<typename GrNode>
struct UpdateRequestCommon {
  GrNode n;
  unsigned int w;
  int id;

UpdateRequestCommon(const GrNode& N, unsigned int W, int ID)
  :n(N), w(W), id(ID)
  {}

  UpdateRequestCommon()
    :n(), w(0)
  {}

  bool operator>(const UpdateRequestCommon& rhs) const {
    return w > rhs.w;
  }

  bool operator<(const UpdateRequestCommon& rhs) const {
    return w < rhs.w;
  }

  bool operator!=(const UpdateRequestCommon& other) const {
    return w != other.w;
  }

  int getID() const {
    return id;
  }
};

struct SNode {
  unsigned int id;
  unsigned int dist;
  
  SNode(int _id = -1) : id(_id), dist(DIST_INFINITY) {}
  std::string toString() {
    std::ostringstream s;
    s << '[' << id << "] dist: " << dist;
    return s.str();
  }
};
