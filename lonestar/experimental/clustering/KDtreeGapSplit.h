#ifndef KD_TREE_GAP_SPLIT_H
#define KD_TREE_GAP_SPLIT_H

/**
 * KD-Tree construction Ideas
 *
 * -- Median split is cheaper than gap split which uses sorting (NlogN). Median
 * split can use std::nth_element(O(N)), which requires random access itertors.
 *    >> We can use Large Array as backing structure, while nodes contain
 * iterators or indices to the Large Array
 *    >> But pushing new items after clustering or unclustered requires a Bag,
 *    which supports forward for bi-directional iterators. Need to copy stuff
 * from bag to large array
 *    >> Per-THread-Vector may be a compromise for this
 *    >> Using LargeArray as backing store for points makes it easy to provide
 *    iterators to go over all points
 *
 * -- Since query nodes for Nearest Neighbor search are all in the tree, NN
 * search may benefit from starting at the node in the tree. Need to investigate
 * it.
 *
 * -- Investigate whether storing a data point at the split point improves
 *  performance
 *
 * -- While parallel clustering algorithm rebuilds the tree every round, serial
 *  algorithm doesn't need to do so, therefore, serial KdTree needs to support
 * add and remove operations
 *
 *
 */

/*
 * Tree building algorithm
 *
 * KdNode<NodeWrapper*> root;
 *
 * wl = { (root, lights.begin(), lights.end())
 *
 * for_each( (node, beg,end) in wl) {
 *    node->buildRecursive(beg, end) {
 *      init(beg, end);
 *      if (end - beg < MAX_POINTS_IN_CELL) {
 *        isLeaf = true;
 *        copy points to local array
 *      } else {
 *        mid = split(beg, end);
 *        l = node->createLeftChild();
 *        r = node->createRightChild();
 *        wl.push( (l, beg, mid) );
 *        wl.push( (r, mid, end) );
 *      }
 *    }
 * }
 */

template <bool CONCURRENT, typename T, typename GetPoint, typename DistFunc>
class KDtree {

public:
  static const unsigned SPLIT_X = 0;
  static const unsigned SPLIT_Y = 1;
  static const unsigned SPLIT_Z = 2;

  static const unsigned MAX_POINTS_IN_CELL = 4;

protected:
  using DataList = galois::LargeArray<T*>;
  using LI       = typename DataList::iterator;

  struct KDcell {
    Point3 m_min;
    Point3 m_max;
    const unsigned m_splitType;
    const double m_splitValue;
    LI m_beg;
    LI m_end;
    KDtree* m_parent;
    KDtree* m_leftChild;
    KDtree* m_rightChild;
  };

  using CellAlloc =
      typename std::conditional<CONCURRENT, galois::FixedSizeAllocator<KDcell>,
                                std::allocator<KDCell>>::type;

  using LeafBag =
      typename std::conditional<CONCURRENT, galois::InsertBag<KDcell*>,
                                std::vector<KDcell*>>::type;

  KDcell* root = nullptr;
  DataList m_dataList;
  LeafBag m_leaves;

public:
  template <typename I>
  typename std::enable_if<CONCURRENT, void>::type build(const I& beg,
                                                        const I& end) {

    size_t sz = std::distance(beg, end);

    m_dataList.allocateBlocked(sz);

    galois::do_all(galois::iterate(0ul, sz), [&](size_t i) {
      m_dataList[i] = *(beg + i); // TODO: implement parallel copy algorithm
    });

          auto it = std::advance(beg
    auto
  }

  template <typename I>
  typename std::enable_if<!CONCURRENT, void>::type build(const I& beg,
                                                         const I& end) {}

  template <typename I, typename WL>
  galois::optional<I> buildRecursive(I beg, I end) {

    auto sz = std::distance(beg, end);

    computeLimits(beg, end);

    if (sz < KDtree::MAX_POINTS_IN_CELL) {
      setLeaf();
      insertPoints(beg, end);
      return galois::optional<I>();

    } else {
      I mid = splitRange(beg, end);
      return galois::optional<I>(mid);
    }
  }

protected:
  template <typename I>
  void computeLimits(const I& beg, const I& end) {

    Point3 min(std::numeric_limits<double>::max());
    Point3 max(-std::numeric_limits<double>::max());

    for (I i = beg; i != end; ++i) {
      min.setIfMin((*i)->getMin());
      max.setIfMax((*i)->getMax());
    }

    m_min = min;
    m_max = max;
  }

  template <typename I>
  I splitRange(I beg, I end) {

    Point3 diff(m_max);
    diff.sub(m_min);

    SplitType splitType0 = SPLIT_X;
    SplitType splitType1 = SPLIT_X;
    SplitType splitType2 = SPLIT_X;

    if (diff.getZ() > diff.getX() && diff.getZ() > diff.getY()) {
      splitType0      = KDtree::SPLIT_Z;
      bool comparCond = diff.getX() > diff.getY();
      splitType1      = comparCond ? KDtree::SPLIT_X : KDtree::SPLIT_Y;
      splitType2      = comparCond ? KDtree::SPLIT_Y : KDtree::SPLIT_X;
    } else if (diff.getY() > diff.getX()) {
      splitType0      = KDtree::SPLIT_Y;
      bool comparCond = diff.getX() > diff.getZ();
      splitType1      = comparCond ? KDtree::SPLIT_X : KDtree::SPLIT_Z;
      splitType2      = comparCond ? KDtree::SPLIT_Z : KDtree::SPLIT_X;
    } else {
      splitType0      = KDtree::SPLIT_X;
      bool comparCond = diff.getY() > diff.getZ();
      splitType1      = comparCond ? KDtree::SPLIT_Y : KDtree::SPLIT_Z;
      splitType2      = comparCond ? KDtree::SPLIT_Z : KDtree::SPLIT_Y;
    }

    SplitType splitTypeUsed = splitType0;
    double splitValueUsed =
        computeSplitValue(list, offset, size, splitType0, arr);
    if (splitValueUsed == std::numeric_limits<double>::max()) {
      splitTypeUsed  = splitType1;
      splitValueUsed = computeSplitValue(list, offset, size, splitType1, arr);
      if (splitValueUsed == std::numeric_limits<double>::max()) {
        splitTypeUsed  = splitType2;
        splitValueUsed = computeSplitValue(list, offset, size, splitType2, arr);
      }
    }
    // Unable to find a good split along any axis!
    if (splitValueUsed == std::numeric_limits<double>::max()) {
      assert(false && "Unable to find a valid split across any dimension!");
    }
  }

  template <SplitType S, typename T = double>
  struct GetSplitComponent {
    T operator()(const NodeWrapper* n) const {
      std::abort();
      return T();
    }
  };

  template <typename T = double>
  struct GetSplitComponent<SPLIT_X, T> {
    T operator()(const NodeWrapper* n) const { return n->getLocationX(); }
  };

  template <typename T = double>
  struct GetSplitComponent<SPLIT_Y, T> {
    T operator()(const NodeWrapper* n) const { return n->getLocationY(); }
  };

  template <typename T = double>
  struct GetSplitComponent<SPLIT_Z, T> {
    T operator()(const NodeWrapper* n) const { return n->getLocationZ(); }
  };

  template <typename I>
  static double computeSplitValue(const I& beg, const I& end,
                                  const SplitType& splitType) {

    switch (splitType) {
    case SPLIT_X:
      return findMedianGapSplit<SPLIT_X>(beg, end) case SPLIT_Y
          : return findMedianGapSplit<SPLIT_Y>(beg, end) case SPLIT_Z
          : return findMedianGapSplit<SPLIT_Z>(beg, end) default : std::abort();
      return 0.0;
    }
  }

  template <typename I, SplitType S>
  static double findMedianGapSplit(const I& beg, const I& end) {

    GetSplitComponent<S, double> getComp;

    auto cmp = [&getComp](const NodeWrapper* a, const NodeWrapper* b) {
      return getComp(a) < getComp(b);
    };

    std::sort(beg, end, cmp);

    auto size = std::distance(beg, end);

    int startOff = ((size - 1) >> 1) - ((size + 7) >> 3);
    int stopOff  = (size >> 1) + ((size + 7) >> 3);
    if (startOff == stopOff) {
      // should never happen
      assert(false && "Start==End in findMedianSplit, should not happen!");
    }

    const auto start = std::advance(beg, startOff);
    const auto stop  = std::advance(beg, stopOff);
    assert(start != stop && "start == stop shouldn't happen");

    double largestGap = 0;
    double splitVal   = 0;

    auto i           = start;
    double nextValue = getComp(*i++);

    for (i != stop; ++i) {

      double curValue = nextValue; // ie val[i]
      nextValue       = getComp(*i);

      if ((nextValue - curValue) > largestGap) {
        largestGap = nextValue - curValue;
        splitVal   = 0.5f * (curValue + nextValue);
        if (splitVal == nextValue) {
          splitVal = curValue;
        } // if not between then choose smaller value
      }
    }
    if (largestGap <= 0) {
      // indicate that the attempt to find a good split value failed
      splitVal = std::numeric_limits<double>::max();
    }
    return splitVal;
  }

  template <typename I>
  static I splitList(const I& beg, const I& end, const SplitType& splitType,
                     const double splitVal) {}

  static KDtree* subDivide(galois::gstl::Vector<NodeWrapper*>& list, int offset,
                           const int size, galois::gstl::Vector<double>* arr,
                           KDtree& factory) {
    KDtree* toReturn;
    if (size <= KDtree::MAX_POINTS_IN_CELL) {

      toReturn     = factory.createNewBlankCell(KDtree::LEAF,
                                            std::numeric_limits<double>::max());
      KDtree& cell = *toReturn;
      for (int i = 0; i < size; i++) {
        cell.m_points[i] = list[offset + i];
      }
      for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
          if (i != j) {
            if (cell.m_points[i]->equals(*cell.m_points[j]))
              assert(false);
          }
        }
      }
      cell.computeBoundingBoxFromPoints(list, size);
      cell.notifyContentsRebuilt(true);
      // TODO: create leaf node
    } else {
      bool shouldClean = false;
      if (arr == NULL) {
        // TODO: create leaf node
        arr         = new galois::gstl::Vector<double>(size);
        shouldClean = true;
      }
      Point3 min(std::numeric_limits<double>::max());
      Point3 max(-std::numeric_limits<double>::max());
      for (int i = offset; i < size; i++) {
        min.setIfMin(list[i]->getMin());
        max.setIfMax(list[i]->getMax());
      }
      Point3 diff(max);
      diff.sub(min);
      int splitTypeUsed     = -1, splitType0, splitType1, splitType2;
      double splitValueUsed = -1;
      if (diff.getZ() > diff.getX() && diff.getZ() > diff.getY()) {
        splitType0      = KDtree::SPLIT_Z;
        bool comparCond = diff.getX() > diff.getY();
        splitType1      = comparCond ? KDtree::SPLIT_X : KDtree::SPLIT_Y;
        splitType2      = comparCond ? KDtree::SPLIT_Y : KDtree::SPLIT_X;
      } else if (diff.getY() > diff.getX()) {
        splitType0      = KDtree::SPLIT_Y;
        bool comparCond = diff.getX() > diff.getZ();
        splitType1      = comparCond ? KDtree::SPLIT_X : KDtree::SPLIT_Z;
        splitType2      = comparCond ? KDtree::SPLIT_Z : KDtree::SPLIT_X;
      } else {
        splitType0      = KDtree::SPLIT_X;
        bool comparCond = diff.getY() > diff.getZ();
        splitType1      = comparCond ? KDtree::SPLIT_Y : KDtree::SPLIT_Z;
        splitType2      = comparCond ? KDtree::SPLIT_Z : KDtree::SPLIT_Y;
      }

      splitTypeUsed  = splitType0;
      splitValueUsed = computeSplitValue(list, offset, size, splitType0, arr);
      if (splitValueUsed == std::numeric_limits<double>::max()) {
        splitTypeUsed  = splitType1;
        splitValueUsed = computeSplitValue(list, offset, size, splitType1, arr);
        if (splitValueUsed == std::numeric_limits<double>::max()) {
          splitTypeUsed = splitType2;
          splitValueUsed =
              computeSplitValue(list, offset, size, splitType2, arr);
        }
      }
      // Unable to find a good split along any axis!
      if (splitValueUsed == std::numeric_limits<double>::max()) {
        assert(false && "Unable to find a valid split across any dimension!");
      }
      int leftCountForSplit =
          splitList(list, offset, size, splitValueUsed, splitTypeUsed);
      if (leftCountForSplit <= 1 || leftCountForSplit >= size - 1) {
        assert(false && "Invalid split");
      }
      toReturn     = factory.createNewBlankCell(splitTypeUsed, splitValueUsed);
      KDtree& cell = *toReturn;
      cell.max.set(max);
      cell.min.set(min);
      cell.m_leftChild =
          subDivide(list, offset, leftCountForSplit, arr, factory);
      cell.m_rightChild = subDivide(list, offset + leftCountForSplit,
                                    size - leftCountForSplit, arr, factory);
      // Clean up on exit.
      if (shouldClean == true)
        delete arr;
    }
    return toReturn;
  }
};

#endif // KD_TREE_GAP_SPLIT_H
