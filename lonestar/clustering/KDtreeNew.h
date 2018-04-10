#ifndef KD_TREE_H
#define KD_TREE_H

/**
 * KD-Tree construction Ideas
 *
 * 1) Accumulate Points in a LargeArray and use that
 * as a backings store for the tree. Each node or leaf nodes have
 * pointers/iterators into the array instead of allocating a small local array
 * sort or split operations move things around in the array itself. 
 * 
 * 1b) Idea 1) can be implemented using Bag with forward iterators or using
 * per-thread-vector and random access iterators. 
 *
 * 2) Use std::nth_element to implement cheap splitting at the median without
 * having to sort etc. Leads to a balanced tree, which may have better parallel
 * performance
 *
 * 3) Investigate whether storing the median split point at internal nodes helps in
 * nearest neighbor searches as opposed to storing split-value and split-type
 *
 * 4) We create different types for Leaf and Internal node. Only Leaf has the array 
 *
 * 5) We keep a null pointer in the node, which is only allocated and filled in the
 * leaf node
 *
 *
 * Minimal Tree interface:
 * -- add range of points (recursive splitting add)
 * -- contains(pt)
 * -- nearestNeighbor(pt)
 * -- recursive delete
 */

template <typename D, typename P>
class KDtree {

public:
  enum SplitType {
    SPLIT_X=0, SPLIT_Y, SPLIT_Z, LEAF
  };
  constexpr static const MAX_POINTS_IN_CELL = 4;

protected:
  using PointList = std::vector<P*>;


  Point3 m_min;
  Point3 m_max;
  const SplitType m_splitType;
  const double m_splitValue;
  KDtree* m_leftChild;
  KDtree* m_rightChild;
  PointList m_pointList;

public:

  KDtree()
      : m_min(std::numeric_limits<double>::max()),
        m_max(-1 * std::numeric_limits<double>::max()), 
        m_splitType(LEAF),
        m_splitValue(std::numeric_limits<double>::max()) 
  {
    m_pointList.resize(MAX_POINTS_IN_CELL);
    m_leftChild      = NULL;
    m_rightChild     = NULL;
  }

  KDtree(int inSplitType, double inSplitValue)
      : min(0), max(0), m_splitType(inSplitType), m_splitValue(inSplitValue) {

    if (m_splitType == LEAF) {
      m_pointList.resize(MAX_POINTS_IN_CELL);
    } else {
      m_pointList.resize(0);
    }

    m_leftChild = m_rightChild = NULL;
  }

  // TODO: simplify
  bool equals(KDtree& other) const {
    if (m_splitType != other.m_splitType) {
      return false;
    }
    if (m_splitValue != other.m_splitValue) {
      return false;
    }
    if (min.equals(other.min) == false) {
      return false;
    }
    if (max.equals(other.max) == false) {
      return false;
    }
    if (m_splitType == KDtree::LEAF) {
      //TODO: create leaf node

      return m_leftChild->equals(*other.m_leftChild) && m_rightChild->equals(*m_rightChild);
    }
    if (m_points.size() != other.m_points.size()) {
      return false;
    }

    for (unsigned int i = 0; i < m_points.size(); i++) {

      if (m_points[i] != NULL && other.m_points[i] != NULL) {
        if (m_points[i]->equals(*other.m_points[i]) == false) {
          return false;
        }
      }
      if (m_points[i] != other.m_points[i]) {
        return false;
      }
    }
    return true;
  }

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

      SplitType splitTypeUsed  = splitType0;
      double splitValueUsed = computeSplitValue(list, offset, size, splitType0, arr);
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

  }

  template <SplitType S, typename T=double>
  struct GetSplitComponent {
     T operator () (const NodeWrapper* n) const { std::abort(); return T(); }
  };

  template <typename T=double>
  struct GetSplitComponent<SPLIT_X, T> {
     T operator () (const NodeWrapper* n) const { return n->getLocationX(); }
  };

  template <typename T=double>
  struct GetSplitComponent<SPLIT_Y, T> {
     T operator () (const NodeWrapper* n) const { return n->getLocationY(); }
  };

  template <typename T=double>
  struct GetSplitComponent<SPLIT_Z, T> {
     T operator () (const NodeWrapper* n) const { return n->getLocationZ(); }
  };

  template <typename I>
  static double computeSplitValue(const I& beg, const I& end, const SplitType& splitType) {

    switch(splitType) {
      case SPLIT_X: 
        return findMedianGapSplit<SPLIT_X>(beg, end)
      case SPLIT_Y: 
        return findMedianGapSplit<SPLIT_Y>(beg, end)
      case SPLIT_Z: 
        return findMedianGapSplit<SPLIT_Z>(beg, end)
      default:
          std::abort(); return 0.0;
    }
  }

  template <typename I, SplitType S>
  static double findMedianGapSplit(const I& beg, const I& end) {

    GetSplitComponent<S, double> getComp;

    auto cmp = [&getComp] (const NodeWrapper* a, const NodeWrapper* b) {
      return getComp(a) < getComp(b);
    };

    std::sort(beg, end, cmp);

    auto size = std::distance(beg, end);

    int startOff = ((size - 1) >> 1) - ((size + 7) >> 3);
    int stopOff   = (size >> 1) + ((size + 7) >> 3);
    if (startOff == stopOff) {
      // should never happen
      assert(false && "Start==End in findMedianSplit, should not happen!");
    }

    const auto start = std::advance(beg, startOff);
    const auto stop = std::advance(beg, stopOff);
    assert(start != stop && "start == stop shouldn't happen");


    double largestGap = 0;
    double splitVal = 0;

    auto i = start;
    double nextValue  = getComp(*i++);

    for (i != stop; ++i) {

      double curValue = nextValue; // ie val[i]
      nextValue       = getComp(*i);

      if ((nextValue - curValue) > largestGap) {
        largestGap = nextValue - curValue;
        splitVal = 0.5f * (curValue + nextValue);
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
  static I splitList(const I& beg, const I& end, const SplitType& splitType, const double splitVal) {
  }

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
      //TODO: create leaf node
    } else {
      bool shouldClean = false;
      if (arr == NULL) {
      //TODO: create leaf node
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
      cell.m_leftChild = subDivide(list, offset, leftCountForSplit, arr, factory);
      cell.m_rightChild = subDivide(list, offset + leftCountForSplit,
                                  size - leftCountForSplit, arr, factory);
      // Clean up on exit.
      if (shouldClean == true)
        delete arr;
    }
    return toReturn;
  }

};


#endif// KD_TREE_H
