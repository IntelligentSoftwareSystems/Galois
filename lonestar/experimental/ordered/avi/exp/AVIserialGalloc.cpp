#include "AVIabstractMain.h"

#include "galois/PriorityQueue.h"


class AVIserialGalloc: public AVIabstractMain {

  virtual const std::string getVersion() const {
    return "Serial version using Galois allocators for PQ";
  }

  virtual void initRemaining (const MeshInit& meshInit, const GlobalVec& g) {}

  virtual void runLoop(MeshInit& meshInit, GlobalVec& g, bool createSyncFiles) {
    // typedef std::priority_queue<AVI*, std::vector<AVI*>, AVIReverseComparator> PQ;
    using PQ = galois::ThreadSafeOrderedSet<AVI*, AVIComparator>; 

    // temporary matrices
    int nrows = meshInit.getSpatialDim ();
    int ncols = meshInit.getNodesPerElem ();

    LocalVec l(nrows, ncols);

    const std::vector<AVI*>& aviList = meshInit.getAVIVec ();

    for (size_t i = 0; i < aviList.size (); ++i) {
      assert (aviList[i]->getOperation ().getFields ().size () == meshInit.getSpatialDim());
    }


    PQ pq;
    for (std::vector<AVI*>::const_iterator i = aviList.begin (), e = aviList.end (); i != e; ++i) {
      pq.push (*i);
      // pq.insert (*i);
    }


    int iter = 0;
    while (!pq.empty ()) {

      AVI* avi = pq.pop();
      // AVI* avi = *pq.begin (); pq.erase (pq.begin ());

      assert (avi != NULL);

      AVIabstractMain::simulate (avi, meshInit, g, l, createSyncFiles);


      if (avi->getNextTimeStamp () < meshInit.getSimEndTime ()) {
        pq.push (avi);
        // pq.insert (avi);
      }

      ++iter;
    }


    // printf ("iterations = %d, time taken (in ms) = %d, average time per iter = %g\n", iter, time, ((double)time)/iter);
    printf ("iterations = %d\n", iter);

  }
};

int main (int argc, char* argv[]) {
  AVIserialGalloc um;
  um.run (argc, argv);
  return 0;
}

