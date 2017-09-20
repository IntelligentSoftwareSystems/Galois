struct DotProductFixedTilingAlgo {
  bool isSgd() const { return false; }
  std::string name() const { return "DotProductFixedTiling"; }
  struct Node {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };
  typedef typename galois::Graph::LC_CSR_Graph<Node, unsigned int>
    ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  void readGraph(Graph& g) { galois::Graph::readGraph(g, inputFilename); }

  void operator()(Graph& g, const StepFunction&) {
    const size_t numUsers = g.size() - NUM_ITEM_NODES;
    const size_t numBlocks0 = (NUM_ITEM_NODES + itemsPerBlock - 1) / itemsPerBlock;
    const size_t numBlocks1 = (numUsers + usersPerBlock - 1) / usersPerBlock;
    const size_t numBlocks = numBlocks0 * numBlocks1;

    std::cout
      << "itemsPerBlock: " << itemsPerBlock
      << " usersPerBlock: " << usersPerBlock
      << " numBlocks: " << numBlocks
      << " numBlocks0: " << numBlocks0
      << " numBlocks1: " << numBlocks1 << "\n";

    typedef typename Graph::GraphNode GNode;
    typedef typename Graph::node_data_type NodeData;
    // computing Root Mean Square Error
    // Assuming only item nodes have edges
    galois::GAccumulator<double> error;
    galois::GAccumulator<size_t> visited;
    galois::Statistic svisited("EdgesVisited");

    constexpr bool useExp = true;
    constexpr bool useDot = false;
    galois::Timer timer;
    galois::runtime::Fixed2DGraphTiledExecutor<Graph, useExp> executor(g, cutoff);
    timer.start();
    executor.execute(
        g.begin(), g.begin() + NUM_ITEM_NODES,
        g.begin() + NUM_ITEM_NODES, g.end(),
        itemsPerBlock, usersPerBlock,
        [&](GNode src, GNode dst, typename Graph::edge_iterator edge) {
      LatentValue e;
      if (useDot) {
        e = predictionError(g.getData(src).latentVector, g.getData(dst).latentVector, g.getEdgeData(edge));
      } else {
        e = doGradientUpdate(g.getData(src).latentVector, g.getData(dst).latentVector, lambda, g.getEdgeData(edge), learningRate);
      }
      error += (e * e);
      visited += 1;
      svisited += 1;
    }, !useDot);
    timer.stop();
    size_t millis = timer.get();
    size_t numVisited = visited.reduce();
    double flop;
    if (useDot) {
      flop = (numVisited * (2.0 * LATENT_VECTOR_SIZE + 2));
    } else {
      flop = countFlops(numVisited, 1, LATENT_VECTOR_SIZE);
    }
    std::cout 
      << "ERROR: " << error.reduce()
      << " Time: " << millis
      << " numIterations: " << numVisited
      << " GFLOP/s: " << flop / millis / 1e6 << "\n";
  }
};

#ifdef HAS_EIGEN

template<Algo algo>
struct AsyncALSalgo {

  bool isSgd() const { return false; }

  std::string name() const { return "AsynchronousAlternatingLeastSquares"; }

  
  struct EmptyBase {};

  typedef typename std::conditional<algo == asyncALSreuse,
          galois::runtime::InputDAGdata,
          EmptyBase>::type NodeBase;

  // TODO: fix compilation when inheriting from NodeBase
  struct Node: public galois::runtime::InputDAGdata { 
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  static const bool NEEDS_LOCKS = algo == asyncALSkdg_i || asyncALSkdg_ar;
  typedef typename galois::Graph::LC_CSR_Graph<Node, unsigned int> BaseGraph;
  typedef typename std::conditional<NEEDS_LOCKS,
          typename BaseGraph::template with_out_of_line_lockable<true>::type,
          typename BaseGraph::template with_no_lockable<true>::type>::type Graph;
  typedef typename Graph::GraphNode GNode;
  // Column-major access 
  typedef Eigen::SparseMatrix<LatentValue> Sp;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, Eigen::Dynamic> MT;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, 1> V;
  typedef Eigen::Map<V> MapV;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, LATENT_VECTOR_SIZE> XTX;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, Eigen::Dynamic> XTSp;

  typedef galois::runtime::PerThreadStorage<XTX> PerThrdXTX;
  typedef galois::runtime::PerThreadStorage<V> PerThrdV;

  Sp A;
  Sp AT;

  void readGraph(Graph& g) {
    galois::Graph::readGraph(g, inputFilename); 
  }

  void copyToGraph(Graph& g, MT& WT, MT& HT) {
    // Copy out
    for (GNode n : g) {
      LatentValue* ptr = &g.getData(n).latentVector[0];
      MapV mapV = { ptr };
      if (n < NUM_ITEM_NODES) {
        mapV = WT.col(n);
      } else {
        mapV = HT.col(n - NUM_ITEM_NODES);
      }
    }
  }

  void copyFromGraph(Graph& g, MT& WT, MT& HT) {
    for (GNode n : g) {
      LatentValue* ptr = &g.getData(n).latentVector[0];
      MapV mapV = { ptr };
      if (n < NUM_ITEM_NODES) {
        WT.col(n) = mapV;
      } else {
        HT.col(n - NUM_ITEM_NODES) = mapV;
      }
    }
  }

  void initializeA(Graph& g) {
    typedef Eigen::Triplet<int> Triplet;
    std::vector<Triplet> triplets { g.sizeEdges() };
    auto it = triplets.begin();
    for (auto n : g) {
      for (auto edge : g.out_edges(n)) {
        *it++ = Triplet(n, g.getEdgeDst(edge) - NUM_ITEM_NODES, g.getEdgeData(edge));
      }
    }
    A.resize(NUM_ITEM_NODES, g.size() - NUM_ITEM_NODES);
    A.setFromTriplets(triplets.begin(), triplets.end());
    AT = A.transpose();
  }

  void visit(Graph& g, size_t col) {
    if (algo == syncALS)
      return;

    g.getData (GNode (col));

    if (col < NUM_ITEM_NODES) {
      for (Sp::InnerIterator it(AT, col); it; ++it)
        g.getData(it.row() + NUM_ITEM_NODES, galois::MethodFlag::READ);
    } else {
      col = col - NUM_ITEM_NODES;
      for (Sp::InnerIterator it(A, col); it; ++it)
        g.getData(it.row(), galois::MethodFlag::READ);
    }
  }

  void update(Graph& g, size_t col, MT& WT, MT& HT,
    PerThrdXTX& xtxs,
    PerThrdV& rhs) 
  {
    // Compute WTW = W^T * W for sparse A
    V& r = *rhs.getLocal();
    // visit(g, col); // TODO: confirm commenting out
    if (col < NUM_ITEM_NODES) {
      r.setConstant(0);
      // HTAT = HT * AT; r = HTAT.col(col)
      for (Sp::InnerIterator it(AT, col); it; ++it)
        r += it.value() * HT.col(it.row());
      XTX& HTH = *xtxs.getLocal();
      HTH.setConstant(0);
      for (Sp::InnerIterator it(AT, col); it; ++it)
        HTH.triangularView<Eigen::Upper>() += HT.col(it.row()) * HT.col(it.row()).transpose();
      for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
        HTH(i, i) += lambda;
      WT.col(col) = HTH.selfadjointView<Eigen::Upper>().llt().solve(r);
    } else {
      col = col - NUM_ITEM_NODES;
      r.setConstant(0);
      // WTA = WT * A; x = WTA.col(col)
      for (Sp::InnerIterator it(A, col); it; ++it)
        r += it.value() * WT.col(it.row());
      XTX& WTW = *xtxs.getLocal();
      WTW.setConstant(0);
      for (Sp::InnerIterator it(A, col); it; ++it)
        WTW.triangularView<Eigen::Upper>() += WT.col(it.row()) * WT.col(it.row()).transpose();
      for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
        WTW(i, i) += lambda;
      HT.col(col) = WTW.selfadjointView<Eigen::Upper>().llt().solve(r);
    }
  }

  struct NonDetTraits { 
    typedef std::tuple<> base_function_traits;
  };

  struct Process
  {
    struct LocalState {
      LocalState(Process&, galois::PerIterAllocTy&) { }
    };

    struct DeterministicId {
      uintptr_t operator()(size_t x) const {
        return x;
      }
    };

    typedef std::tuple<
      galois::needs_per_iter_alloc<>,
      galois::has_intent_to_read<>,
      galois::has_deterministic_local_state<LocalState>,
      galois::has_deterministic_id<DeterministicId>
      > ikdg_function_traits;
    typedef std::tuple<
      galois::needs_per_iter_alloc<>,
      galois::has_fixed_neighborhood<>,
      galois::has_deterministic_local_state<LocalState>,
      galois::has_deterministic_id<DeterministicId>
      > add_remove_function_traits;
    typedef std::tuple<> nondet_function_traits;

    typedef typename std::conditional<(algo != asyncALSkdg_i && algo != asyncALSkdg_ar), 
      nondet_function_traits,
      typename std::conditional<algo == asyncALSkdg_ar,
        add_remove_function_traits,
        ikdg_function_traits>::type>::type function_traits;

    AsyncALSalgo& self;
    Graph& g;
    MT& WT;
    MT& HT;
    PerThrdXTX& xtxs;
    PerThrdV& rhs;

    Process(
      AsyncALSalgo& self,
      Graph& g,
      MT& WT,
      MT& HT,
      PerThrdXTX& xtxs,
      PerThrdV& rhs):
      self(self), g(g), WT(WT), HT(HT), xtxs(xtxs), rhs(rhs) { }
    
    void operator()(size_t col, galois::UserContext<size_t>& ctx) {
      // TODO(ddn) IKDG version can be improved by read/write
      // TODO(ddn) AddRemove can be improevd by reusing DAG
      if (algo == asyncALSkdg_i || algo == asyncALSkdg_ar) {
        // self.visit(g, col);
        if (ctx.isFirstPass()) {
          self.visit(g, col);
          return;
        }
      }
      self.update(g, col, WT, HT, xtxs, rhs);
    }
  };

  // Code for invoking Chromatic and Edge Flipping schedulers
  struct EigenGraphVisitor {

    struct VisitAdjacent {
      Graph& g;
      Sp& A;
      Sp& AT;

      template <typename F>
      void operator () (GNode src, F& func) {
        size_t col = src;
        if (col < NUM_ITEM_NODES) {
          for (Sp::InnerIterator it (AT, col); it; ++it) {
            GNode dst = it.row () + NUM_ITEM_NODES;
            func (dst);
          }

        } else {
          col = col - NUM_ITEM_NODES;
          for (Sp::InnerIterator it (A, col); it; ++it) {
            GNode dst = it.row ();
            func (dst);
          }
        }
      }
    };

    typedef galois::runtime::DAGmanager<Graph, VisitAdjacent> Base_ty;

    struct Manager: public Base_ty {
      Manager (Graph& g, Sp& A, Sp& AT): 
        Base_ty {g, VisitAdjacent {g, A, AT} }
        {}
    };
  };

  struct VisitNhood {
    static const unsigned CHUNK_SIZE = ALS_CHUNK_SIZE;

    AsyncALSalgo& outer;
    Graph& g;

    template <typename C>
    void operator () (size_t col, C&) {
      outer.visit (g, col);
    }
  };

  struct ApplyUpdate {

    static const unsigned CHUNK_SIZE = ALS_CHUNK_SIZE;

    AsyncALSalgo& outer;
    Graph& g;
    MT& WT;
    MT& HT;
    PerThrdXTX& xtxs;
    PerThrdV& rhs;

    template <typename C>
    void operator () (size_t col, C& ctx) {
      (*this) (col);
    }

    void operator () (size_t col) {
      outer.update (g, col, WT, HT, xtxs, rhs);
    }
  };


  void operator()(Graph& g, const StepFunction&) {
    if (!useSameLatentVector && algo != syncALS) {
      galois::runtime::LL::gWarn("Results are not deterministic with different numbers of threads unless -useSameLatentVector is true");
    }
    galois::TimeAccumulator elapsed;
    elapsed.start();

    // Find W, H that minimize ||W H^T - A||_2^2 by solving alternating least
    // squares problems:
    //   (W^T W + lambda I) H^T = W^T A (solving for H^T)
    //   (H^T H + lambda I) W^T = H^T A^T (solving for W^T)
    MT WT { LATENT_VECTOR_SIZE, NUM_ITEM_NODES };
    MT HT { LATENT_VECTOR_SIZE, g.size() - NUM_ITEM_NODES };

    initializeA(g);
    copyFromGraph(g, WT, HT);

    double last = -1.0;
    galois::StatTimer updateTime("UpdateTime");
    galois::StatTimer copyTime("CopyTime");
    PerThrdXTX xtxs;
    PerThrdV rhs;

    typename EigenGraphVisitor::Manager dagManager {g, A, AT}; 

    auto* reuseExec = galois::runtime::make_reusable_dag_exec (
        galois::runtime::makeLocalRange (g),
        g, dagManager, 
        ApplyUpdate {*this, g, WT, HT, xtxs, rhs},
        VisitNhood {*this, g}, std::less<GNode> (),
        "als-async-reuse");

    if (algo == asyncALSreuse) {
      reuseExec->initialize (galois::runtime::makeLocalRange (g));
    }

    for (int round = 1; ; ++round) {

      updateTime.start();

      switch (algo) {
        case syncALS: 
          typedef galois::WorkList::AltChunkedLIFO<ALS_CHUNK_SIZE> WL_ty;
          galois::for_each(
              boost::counting_iterator<size_t>(0),
              boost::counting_iterator<size_t>(NUM_ITEM_NODES),
              Process(*this, g, WT, HT, xtxs, rhs),
              galois::wl<WL_ty> (),
              galois::loopname ("syncALS-users"));
          galois::for_each(
              boost::counting_iterator<size_t>(NUM_ITEM_NODES),
              boost::counting_iterator<size_t>(g.size()),
              Process(*this, g, WT, HT, xtxs, rhs),
              galois::wl<WL_ty> (),
              galois::loopname ("syncALS-movies"));
          break;

        case asyncALSkdg_ar:
        case asyncALSkdg_i:
          galois::for_each(
              boost::counting_iterator<size_t>(0),
              boost::counting_iterator<size_t>(g.size()),
              Process(*this, g, WT, HT, xtxs, rhs),
              galois::wl<galois::WorkList::Deterministic<>>());
          break;

        case asyncALSreuse:
          reuseExec->execute ();
          reuseExec->reinitDAG ();

          break;

        default:
          GALOIS_DIE ("unknown algorithm type");
          break;
      }

      updateTime.stop();

      copyTime.start();
      copyToGraph(g, WT, HT);
      copyTime.stop();

      double error = sumSquaredError(g);
      elapsed.stop();
      std::cout
        << "R: " << round
        << " elapsed (ms): " << elapsed.get()
        << " RMSE (R " << round << "): " << std::sqrt(error/g.sizeEdges())
        << "\n";
      elapsed.start();

      if (fixedRounds <= 0 && round > 1 && std::abs((last - error) / last) < tolerance)
        break;
      if (fixedRounds > 0 && round >= fixedRounds)
        break;

      last = error;
    } // end for

    delete reuseExec;
    reuseExec = nullptr;
  }
};

struct SimpleALSalgo {
  bool isSgd() const { return false; }
  std::string name() const { return "AlternatingLeastSquares"; }
  struct Node { 
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  typedef typename galois::Graph::LC_CSR_Graph<Node, unsigned int>
    ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;
  // Column-major access 
  typedef Eigen::SparseMatrix<LatentValue> Sp;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, Eigen::Dynamic> MT;
  typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, 1> V;
  typedef Eigen::Map<V> MapV;

  Sp A;
  Sp AT;

  void readGraph(Graph& g) {
    galois::Graph::readGraph(g, inputFilename); 
  }

  void copyToGraph(Graph& g, MT& WT, MT& HT) {
    // Copy out
    for (GNode n : g) {
      LatentValue* ptr = &g.getData(n).latentVector[0];
      MapV mapV = { ptr };
      if (n < NUM_ITEM_NODES) {
        mapV = WT.col(n);
      } else {
        mapV = HT.col(n - NUM_ITEM_NODES);
      }
    }
  }

  void copyFromGraph(Graph& g, MT& WT, MT& HT) {
    for (GNode n : g) {
      LatentValue* ptr = &g.getData(n).latentVector[0];
      MapV mapV = { ptr };
      if (n < NUM_ITEM_NODES) {
        WT.col(n) = mapV;
      } else {
        HT.col(n - NUM_ITEM_NODES) = mapV;
      }
    }
  }

  void initializeA(Graph& g) {
    typedef Eigen::Triplet<int> Triplet;
    std::vector<Triplet> triplets { g.sizeEdges() };
    auto it = triplets.begin();
    for (auto n : g) {
      for (auto edge : g.out_edges(n)) {
        *it++ = Triplet(n, g.getEdgeDst(edge) - NUM_ITEM_NODES, g.getEdgeData(edge));
      }
    }
    A.resize(NUM_ITEM_NODES, g.size() - NUM_ITEM_NODES);
    A.setFromTriplets(triplets.begin(), triplets.end());
    AT = A.transpose();
  }

  void operator()(Graph& g, const StepFunction&) {
    galois::TimeAccumulator elapsed;
    elapsed.start();

    // Find W, H that minimize ||W H^T - A||_2^2 by solving alternating least
    // squares problems:
    //   (W^T W + lambda I) H^T = W^T A (solving for H^T)
    //   (H^T H + lambda I) W^T = H^T A^T (solving for W^T)
    MT WT { LATENT_VECTOR_SIZE, NUM_ITEM_NODES };
    MT HT { LATENT_VECTOR_SIZE, g.size() - NUM_ITEM_NODES };
    typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, LATENT_VECTOR_SIZE> XTX;
    typedef Eigen::Matrix<LatentValue, LATENT_VECTOR_SIZE, Eigen::Dynamic> XTSp;
    typedef galois::runtime::PerThreadStorage<XTX> PerThrdXTX;

    initializeA(g);
    copyFromGraph(g, WT, HT);

    double last = -1.0;
    galois::StatTimer mmTime("MMTime");
    galois::StatTimer update1Time("UpdateTime1");
    galois::StatTimer update2Time("UpdateTime2");
    galois::StatTimer copyTime("CopyTime");
    PerThrdXTX xtxs;

    for (int round = 1; ; ++round) {
      mmTime.start();
      // TODO parallelize this using tiled executor
      XTSp WTA = WT * A;
      mmTime.stop();

      update1Time.start();
      galois::for_each(
          boost::counting_iterator<int>(0),
          boost::counting_iterator<int>(A.outerSize()),
          [&](int col, galois::UserContext<int>&) {
        // Compute WTW = W^T * W for sparse A
        XTX& WTW = *xtxs.getLocal();
        WTW.setConstant(0);
        for (Sp::InnerIterator it(A, col); it; ++it)
          WTW.triangularView<Eigen::Upper>() += WT.col(it.row()) * WT.col(it.row()).transpose();
        for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
          WTW(i, i) += lambda;
        HT.col(col) = WTW.selfadjointView<Eigen::Upper>().llt().solve(WTA.col(col));
      });
      update1Time.stop();

      mmTime.start();
      XTSp HTAT = HT * AT;
      mmTime.stop();

      update2Time.start();
      galois::for_each(
          boost::counting_iterator<int>(0),
          boost::counting_iterator<int>(AT.outerSize()),
          [&](int col, galois::UserContext<int>&) {
        // Compute HTH = H^T * H for sparse A
        XTX& HTH = *xtxs.getLocal();
        HTH.setConstant(0);
        for (Sp::InnerIterator it(AT, col); it; ++it)
          HTH.triangularView<Eigen::Upper>() += HT.col(it.row()) * HT.col(it.row()).transpose();
        for (int i = 0; i < LATENT_VECTOR_SIZE; ++i)
          HTH(i, i) += lambda;
        WT.col(col) = HTH.selfadjointView<Eigen::Upper>().llt().solve(HTAT.col(col));
      });
      update2Time.stop();

      copyTime.start();
      copyToGraph(g, WT, HT);
      copyTime.stop();

      double error = sumSquaredError(g);
      elapsed.stop();
      std::cout
        << "R: " << round
        << " elapsed (ms): " << elapsed.get()
        << " RMSE (R " << round << "): " << std::sqrt(error/g.sizeEdges())
        << "\n";
      elapsed.start();

      if (fixedRounds <= 0 && round > 1 && std::abs((last - error) / last) < tolerance)
        break;
      if (fixedRounds > 0 && round >= fixedRounds)
        break;

      last = error;
    }
  }
};
#endif


template<typename Graph, bool UseLocks>
class Recursive2DExecutor {
  typedef galois::runtime::LL::PaddedLock<true> SpinLock;
  typedef typename Graph::GraphNode GNode;
  typedef typename Graph::iterator iterator;
  typedef typename Graph::edge_iterator edge_iterator;
  typedef typename Graph::in_edge_iterator in_edge_iterator;

  template<typename T>
  struct SimpleAtomic {
    std::atomic<T> value;
    SimpleAtomic(): value(0) { }
    SimpleAtomic(const SimpleAtomic& o): value(o.value.load()) { }
    T relaxedLoad() { return value.load(std::memory_order_relaxed); }
    void relaxedAdd(T delta) { value.store(relaxedLoad() + delta, std::memory_order_relaxed); }
  };

  /**
   * Tasks are 2D ranges [start1, end1) x [start2, end2]
   */
  struct Task {
    iterator start1;
    GNode start2;
    iterator end1;
    GNode end2;
    size_t id;
    size_t d1;
    size_t d2;
    SimpleAtomic<size_t> updates;
  };

  Graph& g;
  std::vector<SpinLock> locks1;
  std::vector<SpinLock> locks2;
  std::vector<Task> tasks;
  size_t maxUpdates;
  galois::Statistic failures;
  galois::Statistic successes;
  std::mt19937 gen;

  struct GetDst: public std::unary_function<edge_iterator, GNode> {
    Graph* g;
    GetDst() { }
    GetDst(Graph* _g): g(_g) { }
    GNode operator()(edge_iterator ii) const {
      return g->getEdgeDst(ii);
    }
  };

  struct GetInDst: public std::unary_function<in_edge_iterator, GNode> {
    Graph* g;
    GetInDst() { }
    GetInDst(Graph* _g): g(_g) { }
    GNode operator()(in_edge_iterator ii) const {
      return g->getInEdgeDst(ii);
    }
  };

  typedef galois::NoDerefIterator<edge_iterator> no_deref_iterator;
  typedef galois::NoDerefIterator<in_edge_iterator> no_in_deref_iterator;
  typedef boost::transform_iterator<GetDst, no_deref_iterator> edge_dst_iterator;
  typedef boost::transform_iterator<GetInDst, no_in_deref_iterator> edge_in_dst_iterator;

  template<typename Function>
  void executeBlock(Function& fn, Task& task) {
    constexpr int numTimes = 1;
    constexpr int width = 1;

    GetDst getDst { &g };

    for (auto ii = task.start1; ii != task.end1; ++ii) {
      auto& src = g.getData(*ii);
      edge_iterator begin = g.edge_begin(*ii);
      no_deref_iterator nbegin(begin);
      no_deref_iterator nend(g.edge_end(*ii));
      edge_dst_iterator dbegin(nbegin, getDst);
      edge_dst_iterator dend(nend, getDst);

      for (auto jj = std::lower_bound(dbegin, dend, task.start2); jj != dend; ) {
        bool done = false;
        for (int times = 0; times < numTimes; ++times) {
          for (int i = 0; i < width; ++i) {
            edge_iterator edge = *(jj+i).base();
            if (g.getEdgeDst(edge) > task.end2) {
              done = true;
              break;
            }

            auto& dst = g.getData(g.getEdgeDst(edge));
              
            fn(src, dst, g.getEdgeData(edge));
          }
        }
        if (done)
          break;
        for (int i = 0; jj != dend && i < width; ++jj, ++i)
          ;
        if (jj == dend)
          break;
      }
    }
  }

  template<typename Function>
  void executeLoop(Function fn, unsigned tid, unsigned total) {
    const size_t numBlocks1 = locks1.size();
    const size_t numBlocks2 = locks2.size();
    const size_t numBlocks = numBlocks1 * numBlocks2;
    const size_t block1 = (numBlocks1 + total - 1) / total;
    const size_t start1 = std::min(block1 * tid, numBlocks1 - 1);
    const size_t block2 = (numBlocks2 + total - 1) / total;
    //const size_t start2 = std::min(block2 * tid, numBlocks2 - 1);

    //size_t start = start1 + start2 * numBlocks1; // XXX
    //size_t start = block1 * 10 * (tid / 10) + start2 * numBlocks1;
    size_t start = start1 + block2 * 10 * (tid / 10) * numBlocks1;

    for (int i = 0; ; ++i) {
      start = nextBlock(start, numBlocks, i == 0);
      Task* t = &tasks[start];
      if (t == &tasks[numBlocks])
        break;
      successes += 1;
      executeBlock(fn, *t);

      locks1[t->d1].unlock();
      locks2[t->d2].unlock();
    }
  }

  size_t probeBlock(size_t start, size_t by, size_t n, size_t numBlocks) {
    for (size_t i = 0; i < n; ++i, start += by) {
      while (start >= numBlocks)
        start -= numBlocks;
      Task& b = tasks[start];
      if (b.updates.relaxedLoad() < maxUpdates) {
        if (UseLocks) {
          if (std::try_lock(locks1[b.d1], locks2[b.d2]) < 0) {
            // Return while holding locks
            b.updates.relaxedAdd(1);
            return start;
          }
        } else {
          if (b.updates.value.fetch_add(1) < maxUpdates)
            return start;
        }
      }
    }
    failures += 1;
    return numBlocks;
  }

  // Nested dim1 then dim2
  size_t nextBlock(size_t origStart, size_t numBlocks, bool origInclusive) {
    const size_t delta2 = locks1.size();
    const size_t delta1 = 1;
    size_t b;

    for (int times = 0; times < 2; ++times) {
      size_t limit2 = locks2.size();
      size_t limit1 = locks1.size();
      size_t start = origStart;
      bool inclusive = origInclusive && times == 0;
      // First iteration is exclusive of start
      if ((b = probeBlock(start + (inclusive ? 0 : delta1), delta1, limit1 - (inclusive ? 0 : 1), numBlocks)) != numBlocks)
        return b;
      if ((b = probeBlock(start + (inclusive ? 0 : delta2), delta2, limit2 - (inclusive ? 0 : 1), numBlocks)) != numBlocks)
        return b;
      start += delta1 + delta2;
      while (limit1 > 0 || limit2 > 0) {
        while (start >= numBlocks)
          start -= numBlocks;
        // Subsequent iterations are inclusive of start
        if (limit1 > 0 && (b = probeBlock(start, delta1, limit1 - 1, numBlocks)) != numBlocks)
          return b;
        if (limit2 > 0 && (b = probeBlock(start, delta2, limit2 - 1, numBlocks)) != numBlocks)
          return b;
        if (limit1 > 0) {
          limit1--;
          start += delta1;
        }
        if (limit2 > 0) {
          limit2--;
          start += delta2;
        }
      }
    }

    return numBlocks;
  }

  int countOut(iterator first1, iterator last1, GNode first2, GNode last2) {
    int count = 0;
    GetDst fn { &g };
    for (auto ii = first1; ii != last1; ++ii) {
      edge_dst_iterator start(no_deref_iterator(g.edge_begin(*ii)), fn);
      edge_dst_iterator end(no_deref_iterator(g.edge_end(*ii)), fn);
      for (auto jj = std::lower_bound(start, end, first2); jj != end; ++jj) {
        if (g.getEdgeDst(*jj.base()) > last2)
          break;
        count += 1;
      }
    }
    return count;
  }

  int countIn(iterator first1, iterator last1, GNode first2, GNode last2) {
    int count = 0;
    GetInDst fn { &g };
    for (auto ii = first1; ii != last1; ++ii) {
      edge_in_dst_iterator start(no_in_deref_iterator(g.in_edge_begin(*ii)), fn);
      edge_in_dst_iterator end(no_in_deref_iterator(g.in_edge_end(*ii)), fn);
      for (auto jj = std::lower_bound(start, end, first2); jj != end; ++jj) {
        if (g.getInEdgeDst(*jj.base()) > last2)
          break;
        count += 1;
      }
    }
    return count;
  }

  template<typename It>
  int countHits(It first, It last, GNode a, GNode b, int count) {
    int hits = 0;
    if (first == last)
      return hits;

    std::uniform_int_distribution<size_t> dist(1, last[-1]);
    for (int i = 0; i < count; ++i) {
      size_t v = dist(gen);
      auto ii = std::lower_bound(first, last, v);
      if (std::distance(first, ii) >= a && std::distance(first, ii) < b) {
        hits += 1;
      }
    }
    return hits;
  }

  void initializeTasks(iterator first1, iterator last1, iterator first2, iterator last2, size_t size1, size_t size2) {
    const size_t numBlocks1 = (std::distance(first1, last1) + size1 - 1) / size1;
    const size_t numBlocks2 = (std::distance(first2, last2) + size2 - 1) / size2;
    const size_t numBlocks = numBlocks1 * numBlocks2;

    locks1.resize(numBlocks1);
    locks2.resize(numBlocks2);
    tasks.resize(numBlocks);

    GetDst fn { &g };
    
    double totalSparsity = g.sizeEdges() / ((double) g.size() * g.size());
    // Squared error
    auto se = [](double x, double y) { return (x-y)*(x-y); };
    std::vector<std::array<double, 6>> error;
    for (int i = 0; i < 5; ++i)
      error.emplace_back(std::array<double,6> { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} } );

    std::vector<size_t> sumIn(g.size());
    std::vector<size_t> sumOut(g.size());
    {
      auto pp = sumIn.begin();
      for (auto ii = g.begin(), ei = g.end(); ii != ei; ++ii, ++pp)
        *pp = std::distance(g.in_edge_begin(*ii), g.in_edge_end(*ii));
      std::partial_sum(sumIn.begin(), sumIn.end(), sumIn.begin(), std::plus<size_t>());
    }
    {
      auto pp = sumOut.begin();
      for (auto ii = g.begin(), ei = g.end(); ii != ei; ++ii, ++pp)
        *pp = std::distance(g.edge_begin(*ii), g.edge_end(*ii));
      std::partial_sum(sumOut.begin(), sumOut.end(), sumOut.begin(), std::plus<size_t>());
    }

    for (size_t i = 0; i < numBlocks; ++i) {
      Task& task = tasks[i];
      task.d1 = i % numBlocks1;
      task.d2 = i / numBlocks1;
      task.id = i;
      std::tie(task.start1, task.end1) = galois::block_range(first1, last1, task.d1, numBlocks1);
      // XXX: Works for CSR graphs
      task.start2 = task.d2 * size2 + *first2;
      task.end2 = std::min((task.d2 + 1) * size2 + *first2 - 1, (size_t) last2[-1]);

      int nnz = 0;
      int unique1 = 0;
      std::set<uint64_t> unique2;
      for (auto ii = task.start1; ii != task.end1; ++ii) {
        edge_dst_iterator start(no_deref_iterator(g.edge_begin(*ii)), fn);
        edge_dst_iterator end(no_deref_iterator(g.edge_end(*ii)), fn);
        bool hit = false;
        for (auto jj = std::lower_bound(start, end, task.start2); jj != end; ++jj) {
          if (g.getEdgeDst(*jj.base()) > task.end2)
            break;
          hit = true;
          nnz += 1;
          unique2.insert(g.getEdgeDst(*jj.base()));
        }
        if (hit)
          unique1 += 1;
      }

      ptrdiff_t num1 = std::distance(task.start1, task.end1);
      ptrdiff_t num2 = task.end2 - task.start2;

      // Model 1: uniform sparsity
      {
        double estUnique1 = num1 * (1-std::pow(1-totalSparsity, num2));
        double estUnique2 = num2 * (1-std::pow(1-totalSparsity, num1));
        double estNnz = num1*num2*totalSparsity;
        error[0][0] += se(estUnique1, unique1);
        error[0][1] += se(estUnique2, unique2.size());
        error[0][2] += se(estNnz, nnz);
      }
      // Model 2: Per Node probabilities
      {
        size_t numUsers = g.size() - NUM_ITEM_NODES;
        double density1Min = std::distance(g.edge_begin(*task.start1), g.edge_end(*task.start1)) / (double) NUM_ITEM_NODES;
        double density1Max = std::distance(g.edge_begin(task.end1[-1]), g.edge_end(task.end1[-1])) / (double) NUM_ITEM_NODES;
        double density2Min = std::distance(g.in_edge_begin(task.start2), g.in_edge_end(task.start2)) / (double) numUsers;
        double density2Max = std::distance(g.in_edge_begin(task.end2), g.in_edge_end(task.end2)) / (double) numUsers;
        double estUnique1Min = num1 * (1-std::pow(1-density2Min, num2));
        double estUnique1Max = num1 * (1-std::pow(1-density2Max, num2));
        double estUnique2Min = num2 * (1-std::pow(1-density1Min, num1));
        double estUnique2Max = num2 * (1-std::pow(1-density1Max, num1));
        double estNnzMin = num1*num2*(density1Min+density2Min - density1Min*density2Min);
        double estNnzMax = num1*num2*(density1Max+density2Max - density1Max*density2Max);
        if (true) {
          std::cout << "Model 2: " << task.d1 << "," << task.d2 << "\n";
          std::cout << "1: " << estUnique1Min << " " << estUnique1Max << " " << unique1 << "\n";
          std::cout << "2: " << estUnique2Min << " " << estUnique2Max << " " << unique2.size() << "\n";
          std::cout << "NNZ: " << estNnzMin << " " << estNnzMax << " " << nnz << "\n";
        }
        error[1][0] += se((estUnique1Min+estUnique1Max) / 2, unique1);
        error[1][1] += se((estUnique2Min+estUnique2Max) / 2, unique2.size());
        error[1][2] += se((estNnzMin + estNnzMax) / 2, nnz);
        error[1][3] += estUnique1Min <= unique1 && unique1 <= estUnique1Max ? 0 : 1;
        error[1][4] += estUnique2Min <= unique2.size() && unique2.size() <= estUnique2Max ? 0 : 1;
        error[1][5] += estNnzMin <= nnz && nnz <= estNnzMax ? 0 : 1;
      }
      // Model 3: Interpolate probabilities
      {
        double density1Min = countOut(task.start1, task.start1 + 1, task.start2, task.end2) / (double) std::distance(first1, last1);
        double density1Max = countOut(task.end1 - 1, task.end1, task.start2, task.end2) / (double) std::distance(first1, last1);
        double density2Min = countIn(g.begin()+task.start2, g.begin()+task.start2+1, *task.start1, *task.end1) / (double) std::distance(first2, last2);
        double density2Max = countIn(g.begin()+task.end2-1, g.begin()+task.end2, *task.start1, *task.end1) / (double) std::distance(first2, last2);
        double estUnique1Min = num1 * (1-std::pow(1-density2Min, num2));
        double estUnique1Max = num1 * (1-std::pow(1-density2Max, num2));
        double estUnique2Min = num2 * (1-std::pow(1-density1Min, num1));
        double estUnique2Max = num2 * (1-std::pow(1-density1Max, num1));
        double estNnzMin = num1*num2*(density1Min+density2Min - density1Min*density2Min);
        double estNnzMax = num1*num2*(density1Max+density2Max - density1Max*density2Max);
        if (false) {
          std::cout << "Model 3: " << task.d1 << "," << task.d2 << "\n";
          std::cout << "1: " << estUnique1Min << " " << estUnique1Max << " " << unique1 << "\n";
          std::cout << "2: " << estUnique2Min << " " << estUnique2Max << " " << unique2.size() << "\n";
          std::cout << "NNZ: " << estNnzMin << " " << estNnzMax << " " << nnz << "\n";
        }
        error[2][0] += se((estUnique1Min+estUnique1Max) / 2, unique1);
        error[2][1] += se((estUnique2Min+estUnique2Max) / 2, unique2.size());
        error[2][2] += se((estNnzMin + estNnzMax) / 2, nnz);
        error[2][3] += estUnique1Min <= unique1 && unique1 <= estUnique1Max ? 0 : 1;
        error[2][4] += estUnique2Min <= unique2.size() && unique2.size() <= estUnique2Max ? 0 : 1;
        error[2][5] += estNnzMin <= nnz && nnz <= estNnzMax ? 0 : 1;
        
      }
      // Model 4: null
      {
        error[3][0] += se(0, unique1);
        error[3][1] += se(0, unique2.size());
        error[3][2] += se(0, nnz);
      }
      // Model 5: Sample in-degree distribution
      {
        double density1 = countHits(sumIn.begin(), sumIn.end(), task.start2, task.end2, 1000) / (double) 1000;
        double density2 = countHits(sumOut.begin(), sumOut.end(), *task.start1, *task.end1, 1000) / (double) 1000;
        double estUnique1 = num1 * (1-std::pow(1-density2, num2));
        double estUnique2 = num2 * (1-std::pow(1-density1, num1));
        double estNnz = num1*num2*(density1+density2 - density1*density2);
        if (false) {
          std::cout << "Model 4: " << task.d1 << "," << task.d2 << "\n";
          std::cout << "1: " << estUnique1 << " " << unique1 << "\n";
          std::cout << "2: " << estUnique2 << " " << unique2.size() << "\n";
          std::cout << "NNZ: " << estNnz << " " << nnz << "\n";
        }
        error[4][0] += se(estUnique1, unique1);
        error[4][1] += se(estUnique2, unique2.size());
        error[4][2] += se(estNnz, nnz);
        
      }
    }
    for (size_t i = 0; i < error.size(); ++i) {
      std::cout 
        << "RMSE Model " << i << ":" 
        << " 1: " << std::sqrt(error[i][0] / numBlocks) << " (" << error[i][3] / numBlocks << ")"
        << " 2: " << std::sqrt(error[i][1] / numBlocks) << " (" << error[i][4] / numBlocks << ")"
        << " NNZ: " << std::sqrt(error[i][2] / numBlocks) << " (" << error[i][5] / numBlocks << ")"
        << "\n";
    }
  }

  template<typename Function>
  struct Process {
    Recursive2DExecutor* self;
    Function fn;

    void operator()(unsigned tid, unsigned total) {
      self->executeLoop(fn, tid, total);
    }
  };

public:
  Recursive2DExecutor(Graph& _g): g(_g), failures("PopFailures"), successes("PopSuccesses") { }

  template<typename Function>
  size_t execute(iterator first1, iterator last1, iterator first2, iterator last2, size_t size1, size_t size2, Function fn, size_t numIterations = 1) {
    galois::Timer timer;
    timer.start();
    initializeTasks(first1, last1, first2, last2, size1, size2);
    timer.stop();
    maxUpdates = numIterations;
    Process<Function> p = { this, fn };
    galois::on_each(p);
    return timer.get();
  }
};

struct DotProductRecursiveTilingAlgo {
  bool isSgd() const { return false; }
  std::string name() const { return "DotProductRecursiveTiling"; }
  struct Node {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  typedef typename galois::Graph::LC_CSR_Graph<Node, unsigned int>
//    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type InnerGraph;
  typedef typename galois::Graph::LC_InOut_Graph<InnerGraph> Graph;
  typedef Graph::GraphNode GNode;

  void readGraph(Graph& g) { galois::Graph::readGraph(g, inputFilename, transposeGraphName); }

  struct GetDistance: public std::unary_function<GNode, ptrdiff_t> {
    Graph* g;
    GetDistance() { }
    GetDistance(Graph* _g): g(_g) { }
    ptrdiff_t operator()(GNode x) const {
      if (g->edge_begin(x) == g->edge_end(x))
        return std::distance(g->in_edge_begin(x), g->in_edge_end(x));
      return std::distance(g->edge_begin(x), g->edge_end(x));
    }
  };

  void operator()(Graph& g, const StepFunction&) {
    typedef typename Graph::GraphNode GNode;
    typedef typename Graph::node_data_type NodeData;
    typedef boost::transform_iterator<GetDistance, typename Graph::iterator> distance_iterator;

    galois::Timer timer;
    timer.start();
    galois::GAccumulator<double> error;
    galois::GAccumulator<size_t> visited;
    Recursive2DExecutor<Graph,false> executor(g);
    distance_iterator start1(g.begin(), GetDistance(&g));
    distance_iterator end1(g.begin()+NUM_ITEM_NODES, GetDistance(&g));
    distance_iterator start2(g.begin()+NUM_ITEM_NODES, GetDistance(&g));
    distance_iterator end2(g.end(), GetDistance(&g));

    auto first1 = start1.base();
    if (cutoff > 0)
      first1 = std::lower_bound(start1, end1, std::abs(cutoff)).base();
    auto last1 = end1.base();
    if (cutoff < 0)
      last1 = std::upper_bound(start1, end1, std::abs(cutoff)).base();

    auto first2 = start2.base();
    if (cutoff > 0)
      first2 = std::lower_bound(start2, end2, std::abs(cutoff)).base();
    auto last2 = end2.base();
    if (cutoff < 0)
      last2 = std::upper_bound(start2, end2, std::abs(cutoff)).base();

    size_t inspectTime = executor.execute(first1, last1, first2, last2,
        itemsPerBlock, usersPerBlock,
        [&](NodeData& nn, NodeData& mm, unsigned int edgeData) {
      LatentValue e = predictionError(nn.latentVector, mm.latentVector, edgeData);

      error += (e * e);
      visited += 1;
    });
    timer.stop();
    std::cout 
      << "ERROR: " << error.reduce()
      << " Time: " << timer.get() - inspectTime
      << " Iterations: " << visited.reduce()
      << " GFLOP/s: " << (visited.reduce() * (2.0 * LATENT_VECTOR_SIZE + 2)) / (timer.get() - inspectTime) / 1e6 << "\n";
  }
};


struct BlockJumpAlgo {
  bool isSgd() const { return true; }
  typedef galois::runtime::LL::PaddedLock<true> SpinLock;
  static const bool precomputeOffsets = false;

  std::string name() const { return "BlockAlgo"; }

  struct Node {
    LatentValue latentVector[LATENT_VECTOR_SIZE];
  };

  typedef galois::Graph::LC_CSR_Graph<Node, unsigned int>
//    ::with_numa_alloc<true>::type
    ::with_no_lockable<true>::type Graph;
  typedef Graph::GraphNode GNode;

  void readGraph(Graph& g) { galois::Graph::readGraph(g, inputFilename); }

  size_t userIdToUserNode(size_t userId) {
    return userId + NUM_ITEM_NODES;
  }

  struct BlockInfo {
    size_t id;
    size_t x;
    size_t y;
    size_t userStart;
    size_t userEnd;
    size_t itemStart;
    size_t itemEnd;
    size_t numitems;
    size_t updates;
    double error;
    int* userOffsets;

    std::ostream& print(std::ostream& os) {
      os 
        << "id: " << id
        << " x: " << x
        << " y: " << y
        << " userStart: " << userStart
        << " userEnd: " << userEnd
        << " itemStart: " << itemStart
        << " itemEnd: " << itemEnd
        << " updates: " << updates
        << "\n";
      return os;
    }
  };

  struct Process {
    Graph& g;
    SpinLock *xLocks, *yLocks;
    BlockInfo* blocks;
    size_t numXBlocks, numYBlocks;
    LatentValue* steps;
    size_t maxUpdates;
    galois::GAccumulator<double>* errorAccum;
    
    struct GetDst: public std::unary_function<Graph::edge_iterator, GNode> {
      Graph* g;
      GetDst() { }
      GetDst(Graph* _g): g(_g) { }
      GNode operator()(Graph::edge_iterator ii) const {
        return g->getEdgeDst(ii);
      }
    };

    /**
     * Preconditions: row and column of slice are locked.
     *
     * Postconditions: increments update count, does sgd update on each item
     * and user in the slice
     */
    template<bool Enable = precomputeOffsets>
    size_t runBlock(BlockInfo& si, typename std::enable_if<!Enable>::type* = 0) {
      typedef galois::NoDerefIterator<Graph::edge_iterator> no_deref_iterator;
      typedef boost::transform_iterator<GetDst, no_deref_iterator> edge_dst_iterator;

      LatentValue stepSize = steps[si.updates - maxUpdates + updatesPerEdge];
      size_t seen = 0;
      double error = 0.0;

      // Set up item iterators
      size_t itemId = 0;
      Graph::iterator mm = g.begin(), em = g.begin();
      std::advance(mm, si.itemStart);
      std::advance(em, si.itemEnd);
      
      GetDst fn { &g };

      // For each item in the range
      for (; mm != em; ++mm, ++itemId) {  
        GNode item = *mm;
        Node& itemData = g.getData(item);
        size_t lastUser = si.userEnd + NUM_ITEM_NODES;

        edge_dst_iterator start(no_deref_iterator(g.edge_begin(item, galois::MethodFlag::UNPROTECTED)), fn);
        edge_dst_iterator end(no_deref_iterator(g.edge_end(item, galois::MethodFlag::UNPROTECTED)), fn);

        // For each edge in the range
        for (auto ii = std::lower_bound(start, end, si.userStart + NUM_ITEM_NODES); ii != end; ++ii) {
          GNode user = g.getEdgeDst(*ii.base());

          if (user >= lastUser)
            break;

          LatentValue e = doGradientUpdate(itemData.latentVector, g.getData(user).latentVector, lambda, g.getEdgeData(*ii.base()), stepSize);
          if (errorAccum)
            error += e * e;
          ++seen;
        }
      }

      si.updates += 1;
      if (errorAccum) {
        *errorAccum += (error - si.error);
        si.error = error;
      }
      
      return seen;
    }

    template<bool Enable = precomputeOffsets>
    size_t runBlock(BlockInfo& si, typename std::enable_if<Enable>::type* = 0) {
      LatentValue stepSize = steps[si.updates - maxUpdates + updatesPerEdge];
      size_t seen = 0;
      double error = 0.0;

      // Set up item iterators
      size_t itemId = 0;
      Graph::iterator mm = g.begin(), em = g.begin();
      std::advance(mm, si.itemStart);
      std::advance(em, si.itemEnd);
      
      // For each item in the range
      for (; mm != em; ++mm, ++itemId) {  
        if (si.userOffsets[itemId] < 0)
          continue;

        GNode item = *mm;
        Node& itemData = g.getData(item);
        size_t lastUser = si.userEnd + NUM_ITEM_NODES;

        // For each edge in the range
        for (auto ii = g.edge_begin(item) + si.userOffsets[itemId], ei = g.edge_end(item); ii != ei; ++ii) {
          GNode user = g.getEdgeDst(ii);

          if (user >= lastUser)
            break;

          LatentValue e = doGradientUpdate(itemData.latentVector, g.getData(user).latentVector, lambda, g.getEdgeData(ii), stepSize);
          if (errorAccum)
            error += e * e;
          ++seen;
        }
      }

      si.updates += 1;
      if (errorAccum) {
        *errorAccum += (error - si.error);
        si.error = error;
      }
      
      return seen;
    }

    /**
     * Searches next slice to work on.
     *
     * @returns slice id to work on, x and y locks are held on the slice
     */
    size_t getNextBlock(BlockInfo* sp) {
      size_t numBlocks = numXBlocks * numYBlocks;
      size_t nextBlockId = sp->id + 1;
      for (size_t i = 0; i < 2 * numBlocks; ++i, ++nextBlockId) {
        // Wrap around
        if (nextBlockId == numBlocks)
          nextBlockId = 0;

        BlockInfo& nextBlock = blocks[nextBlockId];

        if (nextBlock.updates < maxUpdates && xLocks[nextBlock.x].try_lock()) {
          if (yLocks[nextBlock.y].try_lock()) {
            // Return while holding locks
            return nextBlockId;
          } else {
            xLocks[nextBlock.x].unlock();
          }
        }
      }

      return numBlocks;
    }

    void operator()(unsigned tid, unsigned total) {
      galois::StatTimer timer("PerThreadTime");
      galois::Statistic edgesVisited("EdgesVisited");
      galois::Statistic blocksVisited("BlocksVisited");
      size_t numBlocks = numXBlocks * numYBlocks;
      size_t xBlock = (numXBlocks + total - 1) / total;
      size_t xStart = std::min(xBlock * tid, numXBlocks - 1);
      size_t yBlock = (numYBlocks + total - 1) / total;
      size_t yStart = std::min(yBlock * tid, numYBlocks - 1);
      BlockInfo* sp = &blocks[xStart + yStart + numXBlocks];

      timer.start();

      while (true) {
        sp = &blocks[getNextBlock(sp)];
        if (sp == &blocks[numBlocks])
          break;
        blocksVisited += 1;
        edgesVisited += runBlock(*sp);

        xLocks[sp->x].unlock();
        yLocks[sp->y].unlock();
      }

      timer.stop();
    }
  };

  void operator()(Graph& g, const StepFunction& sf) {
    const size_t numUsers = g.size() - NUM_ITEM_NODES;
    const size_t numYBlocks = (NUM_ITEM_NODES + itemsPerBlock - 1) / itemsPerBlock;
    const size_t numXBlocks = (numUsers + usersPerBlock - 1) / usersPerBlock;
    const size_t numBlocks = numXBlocks * numYBlocks;

    SpinLock* xLocks = new SpinLock[numXBlocks];
    SpinLock* yLocks = new SpinLock[numYBlocks];
    
    std::cout
      << "itemsPerBlock: " << itemsPerBlock
      << " usersPerBlock: " << usersPerBlock
      << " numBlocks: " << numBlocks
      << " numXBlocks: " << numXBlocks
      << " numYBlocks: " << numYBlocks << "\n";
    
    // Initialize
    BlockInfo* blocks = new BlockInfo[numBlocks];
    for (size_t i = 0; i < numBlocks; i++) {
      BlockInfo& si = blocks[i];
      si.id = i;
      si.x = i % numXBlocks;
      si.y = i / numXBlocks;
      si.updates = 0;
      si.error = 0.0;
      si.userStart = si.x * usersPerBlock;
      si.userEnd = std::min((si.x + 1) * usersPerBlock, numUsers);
      si.itemStart = si.y * itemsPerBlock;
      si.itemEnd = std::min((si.y + 1) * itemsPerBlock, NUM_ITEM_NODES);
      si.numitems = si.itemEnd - si.itemStart;
      if (precomputeOffsets) {
        si.userOffsets = new int[si.numitems];
      } else {
        si.userOffsets = nullptr;
      }
    }

    // Partition item edges in blocks to users according to range [userStart, userEnd)
    if (precomputeOffsets) {
      galois::do_all(g.begin(), g.begin() + NUM_ITEM_NODES, [&](GNode item) {
        size_t sliceY = item / itemsPerBlock;
        BlockInfo* s = &blocks[sliceY * numXBlocks];

        size_t pos = item - s->itemStart;
        auto ii = g.edge_begin(item), ei = g.edge_end(item);
        size_t offset = 0;
        for (size_t i = 0; i < numXBlocks; ++i, ++s) {
          size_t start = userIdToUserNode(s->userStart);
          size_t end = userIdToUserNode(s->userEnd);

          if (ii != ei && g.getEdgeDst(ii) >= start && g.getEdgeDst(ii) < end) {
            s->userOffsets[pos] = offset;
          } else {
            s->userOffsets[pos] = -1;
          }
          for (; ii != ei && g.getEdgeDst(ii) < end; ++ii, ++offset)
            ;
        }
      });
    }
    
    executeUntilConverged(sf, g, [&](LatentValue* steps, size_t maxUpdates, galois::GAccumulator<double>* errorAccum) {
      Process fn { g, xLocks, yLocks, blocks, numXBlocks, numYBlocks, steps, maxUpdates, errorAccum };
      galois::on_each(fn);
    });
  }
};
