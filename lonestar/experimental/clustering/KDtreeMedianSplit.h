#ifndef KD_TREE_MEDIAN_SPLIT_H
#define KD_TREE_MEDIAN_SPLIT_H

template <bool CONCURRENT, typename T, typename GetPointFunc, typename DistFunc>
class KDtree {

public:
  static const unsigned SPLIT_X = 0;
  static const unsigned SPLIT_Y = 1;
  static const unsigned SPLIT_Z = 2;

  static const unsigned MAX_POINTS_IN_CELL = 8;

protected:

  using DataArray = galois::LargeArray<T*>;
  using AI = typename DataArray::iterator;
 
  struct KDcell {
    unsigned mSplitType;
    double mSplitVal;
    AI mBeg;
    AI mEnd;
    KDtree* mLeftChild;
    KDtree* mRightChild;
  };

  using CellAlloc = typename std::conditional<CONCURRENT, 
        galois::FixedSizeAllocator<KDcell>, std::allocator<KDCell> >::type;


  using PushBag = galois::PerThreadBag<T*>;

  GetPointFunc mGetPtFunc;
  DistFunc mDistFunc;

  KDcell* mRoot;

  PushBag mNextBag;
  DataArray mDataArray;
  CellAlloc mCellAlloc;

  KDcell* createCell(AI beg, AI end {
    KDcell* node = mCellAlloc.allocate(1);
    assert(node);
    mCellAlloc.construct(node, SPLIT_X, 0.0, beg, end, nullptr, nullptr);
  }

public:

  explicit KDtree(const GetPointFunc& getPtFunc=GetPointFunc(), const DistFunc& distFunc=DistFunc())
    :
      mGetPtFunc(getPtFunc),
      mDistFunc(distFunc),
      mRoot(nullptr)
  {}


  void pushNext(T* item) {
    mNextBag.push(item);
  }

  void buildNext(void) {

    copyBagToArray(); // TODO: implement

    KDcell* root = createCell(mDataArray.begin(), mDataArray.end());
    mRoot = node;

    galois::for_each(galois::iterate({root}),
        [&] (KDcell* cell, auto& ctx) {

          auto sz = std::distance(cell->mBeg, cell->mEnd);

          if (sz > MAX_POINTS_IN_CELL) {



            KDcell* left = createCell(...);
          }
        },
        galois::no_aborts());


    
  }



};

#endif// KD_TREE_MEDIAN_SPLIT_H
