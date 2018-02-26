#ifndef _BCGRAPH_H_
#define _BCGRAPH_H_

#include "galois/Bag.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>

#ifdef __linux__
#include <linux/mman.h>
#endif

#include <cerrno>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <list>

#undef STOREOUTNBRS 

//
//  XXX Important assumption about the graphs read: Node id's start at 0 and 
//  go up to nnodes - 1
//
class BCGraph {
 public:
 //private:
  galois::substrate::CacheLineStorage<ND>* nodes;
  ED* edgeData;
  int* inIdx;
  int* ins;
  int* outIdx;
  #ifdef STOREOUTNBRS
  int* outs;
  int nouts;
  #endif
  int nnodes;
  int nedges;
  int ninIdx;
  int nins;
  int noutIdx;

  void* masterMapping;
  size_t masterLength;
  int masterFD;

  /**
   * mmaps the provided file.
   *
   * @param filename file to mmap
   * @returns a tuple containing a pointer to the mmapped file, the filesize,
   * and the file descriptor (or a nullptr if mmap failed
   */
  boost::tuple<void*, int, int> mapFile(const char *filename) {
    int fileFD = open(filename, O_RDONLY);
    struct stat buf;
    fstat(fileFD, &buf);
    size_t fileLength = buf.st_size;

    int _MAP_BASE = MAP_PRIVATE /*| MAP_HUGETLB*/;

    #ifdef MAP_POPULATE
    galois::gInfo("We have map_populate");
    _MAP_BASE  |= MAP_POPULATE;
    #endif

    void* m = mmap(0, fileLength, PROT_READ,_MAP_BASE, fileFD, 0);

    if (m == MAP_FAILED) {
      m = 0;
      galois::gError("Problem with mmap ", filename);
      perror("ERROR: ");
    }

    galois::gInfo("mmaped file ", filename, " of size ", fileLength);

    return boost::make_tuple(m, fileLength, fileFD);
  }

 public:
  BCGraph(const char* filename) {
    galois::gInfo("Node Size ", sizeof(ND), " uses ", 
                  sizeof(galois::substrate::CacheLineStorage<ND>));
    galois::gInfo("Edge Size ", sizeof(ED));
    std::string tmp(filename);
    std::string fname1 = tmp + "1.gr";
    std::string fname2 = tmp + "2.gr"; 

    boost::tuple<void*, int, int> ft1 = mapFile(fname1.c_str());
   
    void *m = ft1.get<0>();
    masterMapping = m;
    masterLength = ft1.get<1>();
    masterFD = ft1.get<2>();

    uint64_t* fptr = (uint64_t*)m;
    uint64_t version = *fptr++;
    assert(version == 2);
    uint64_t numNodes = *fptr++;
    uint64_t numEdges = *fptr++;
    galois::gInfo("Read numNodes ", numNodes, " numEdges: ", numEdges); 
    
    nnodes = numNodes;
    nedges = numEdges;
    ninIdx = nnodes+1;
    nins = nedges;
    noutIdx = ninIdx;
    
    int idxArraysLen = numNodes + 1;

    uint32_t* fptr32 = (uint32_t*)fptr;

    inIdx = (int*)fptr32;
    fptr32 += idxArraysLen;
    if (idxArraysLen % 2) fptr32++;
    ins = (int *)fptr32; 
    fptr32 += numEdges;
    if (numEdges % 2) fptr32 += 1;
    outIdx = (int *)fptr32;
    fptr32 += idxArraysLen;
    if (idxArraysLen % 2) fptr32++;

    boost::tuple<void*,int,int> ft2 = mapFile(fname2.c_str());
    void *m2 = ft2.get<0>();
    fptr = (uint64_t*)m2;
    version = *fptr++;
    assert(version == 2);
    fptr32 = (uint32_t*)fptr;

    int* outs = (int *)fptr32; 
    fptr32 += numEdges;
    if (numEdges % 2) fptr32 += 1;
    
    nodes = new galois::substrate::CacheLineStorage<ND>[nnodes];
    for (int i =0; i<nnodes; ++i) {
      nodes[i].data.id = i;
    }

    edgeData = new ED[nedges];
    for (int i=0; i<nnodes; ++i) {
      int start = outIdx[i];
      int end = outIdx[i+1];

      for (int j = start; j < end; j++) {
        int nbr = outs[j];
        ED & e = edgeData[j];
        e.src = &(nodes[i].data);
        e.dst = &(nodes[nbr].data);
      } 
    }

    munmap(m2, ft2.get<1>());
    close(ft2.get<2>());
    
    #if defined (__SVR4) && defined (__sun)
    int sum = 0;
    std::cerr << "BEFORE DUMMY: " << sum << std::endl;
    for (int k=0; k<nins; ++k) {
      sum += ins[k];
    }
    for (int k=0; k<ninIdx; ++k) {
      sum += inIdx[k];
    }
    std::cerr << "DUMMY: " << sum << std::endl;
    #endif
  }

  ~BCGraph() {
    if (masterMapping) munmap(masterMapping, masterLength);
    if (masterFD) close(masterFD);
  }

  #define CONSTR_DBG 0

  int size() const {
    return nnodes;
  }

  int getNedges() const {
    return nedges;
  }
  
  ND* getNode(int id) {
    return &nodes[id].data;
  }

  galois::substrate::CacheLineStorage<ND>* getNodes() const {
    return nodes;
  }
 
  template<typename Context>
  void /*__attribute__((noinline))*/ inline addOutEdgesToWL(ND *n, 
                                                            /*int ndist,*/ 
                                                            Context & ctx) {
    int idx = n->id;
    assert(idx <= nnodes-1);
    int start = outIdx[idx];
    int end = outIdx[idx + 1];
    for (int i = start; i < end; i++) {
      assert(i<=nedges-1);
      ED & e = edgeData[i];
      assert (e.src == n);
      if (/*e.level >= ndist e.dst->distance >= ndist &&*/ n != e.dst) {
//        if (!e.isAlreadyIn())
          ND *outNbr = e.dst;
          ND * aL;
          ND *aW;
          if (n < outNbr) { aL = n; aW = outNbr;} 
          else { aL = outNbr; aW = n; }
          aL->lock(); aW->lock();
          if (outNbr->distance > n->distance) {
            ctx.push(&e);
          }
          aL->unlock(); aW->unlock();
      }
    }
  }

  template<typename Context>
  void /*__attribute__((noinline))*/ inline addOutEdgesToWL2(ND *n, 
                                                             int ndist, 
                                                             Context& ctx) {
    int idx = n->id;
    assert(idx <= nnodes-1);
    int start = outIdx[idx];
    int end = outIdx[idx + 1];
    for (int i = start; i < end; i++) {
      assert(i<=nedges-1);
      ED & e = edgeData[i];
      assert (e.src == n);
      if (/*e.level >= ndist e.dst->distance >= ndist &&*/ n != e.dst) {
        if (!e.isAlreadyIn()) ctx.push(&e);
/*          ND *outNbr = e.dst;
          ND * aL;
          ND *aW;
          if (n < outNbr) { aL = n; aW = outNbr;} 
          else { aL = outNbr; aW = n; }
          aL->lock(); aW->lock();
          if (n->distance != ndist) {
            aL->unlock(); aW->unlock();
            break;
          }
          if (outNbr->distance > n->distance) {
            ctx.push(&e);
          }
          aL->unlock(); aW->unlock();*/
      }
    }
  }



  template<typename Context>
  void /* __attribute__((noinline))*/ inline addInEdgesToWL(ND *n, 
      /*int ndist,*/ Context & ctx, ED* exception) {
    int idx = n->id;
    int start = inIdx[idx];
    int end = inIdx[idx + 1];
    for (int i = start; i < end; i++) {
      //if (i>= nedges) {
      //  std::cerr << "i=" << i << " idx: " << idx << " start: " << start << " end: " << end << std::endl;
      //}
      assert(i<nedges);
      assert(ins[i]<nedges);
      ED & e = edgeData[ins[i]];
      /*if (e->dst != n) {
        std::cerr << "Problem: " << e->dst->toString() << " ||| " << n->toString() << " || " << e->toString() << std::endl;
      }*/
      assert (e.dst == n);
      if (/*e.level <= ndist && */ &e != exception && e.src != n/*e.dst*/) {
        if (!e.isAlreadyIn())
          ctx.push(&e);
      }
    }
  }

  //void initWLToOutEdges(ND *n, std::list<ED*> & wl) {
  //void initWLToOutEdges(ND *n, std::vector<ED*> & wl) {
  void initWLToOutEdges(ND *n, galois::InsertBag<ED*> & wl) {
    int idx = n->id;
    int start = outIdx[idx];
    int end = outIdx[idx + 1];
    for (int i = start; i < end; i++) {
      assert(i<nedges);
      ED & e = edgeData[i];
      if (n != e.dst) {
        wl.push_back(&e);
      }
    }
  }

  void resetOutEdges(ND *n) {
    int idx = n->id;
    int start = outIdx[idx];
    int end = outIdx[idx + 1];
    for (int i = start; i < end; i++) {
      assert(i<nedges);
      ED & e = edgeData[i];
      e.reset();
    }
  }

  int neighborsSize(const ND *src, const int* const adjIdx) const {
    int idx = src->id;
    assert(idx<nnodes);
    int start = adjIdx[idx];
    int end = adjIdx[idx + 1];
    return end - start;
  }

  void fixNodePredsCapacities() {
    //int maxInNbrs = 0;
    //int maxId;
    //int sumInNbrs = 0;
    //int maxOutNbrs = 0;
    ////#if 0
    //for (int i=0; i<nnodes; ++i) {
    //  ND & n = nodes[i].data;
    //  int nOutNbrs = inNeighborsSize(&n);
    //  //n.preds.reserve(std::min(2, nOutNbrs));
    //  if (maxInNbrs < nOutNbrs) {
    //    maxInNbrs = nOutNbrs;
    //    maxId = n.id;
    //  }
    //  sumInNbrs += nOutNbrs;
    //  int oos = outNeighborsSize(&n);
    //  if (maxOutNbrs < oos) {
    //    maxOutNbrs = oos;
    //  }
    //}
    ////#endif
    //std::cerr << "Node " << maxId << " has " << maxInNbrs << " in-nbrs Sum is " << sumInNbrs << "\n";
    //std::cerr << " Max " << maxOutNbrs << " out-nbrs \n";
  }

  void fixNodepredsCapacities(int start, int end) {
    #if 0
    for (int i=start; i<end; ++i) {
      ND & n = nodes[i].data;
      int nOutNbrs = inNeighborsSize(&n);
      //n.preds.reserve(std::min(2, nOutNbrs));
    }
    #endif
  }

  int inline inNeighborsSize(const ND *src) {
    return neighborsSize(src, inIdx);
  }

  int outNeighborsSize(const ND *src) {
    return neighborsSize(src, outIdx);
  }

  void inline cleanupData(/*int nstart, int nend, */int edgStart, int edgEnd) {
  /*    for (int j=nstart; j<nend; j++) {   
       assert(j<nnodes);
       nodes[j].data.reset();
     }*/
    for (int j=edgStart; j<edgEnd; ++j) {
      assert(j<nedges);
      edgeData[j].reset();
    }
  }
 
  void checkClearEdges() {
    for (int j=0; j<nedges; ++j) edgeData[j].checkClear(j);
  }

  void cleanupData() {
    for (int j=0; j<nnodes; j++) { 
      assert(j<nnodes);
      nodes[j].data.reset();
    }

    for (int j=0; j<nedges; ++j) {
      assert(j<nedges);
      edgeData[j].reset();
    }
  }

  /*
  void cleanupDataOMP() {
    #pragma omp parallel 
    {
      int nthreads = omp_get_num_threads();
      std::cerr << "OMP Threads: " << nthreads << std::endl;
      int end = nnodes;
      int chunk = nnodes/nthreads;
      #pragma omp for schedule(static, chunk) private(end)
      for (int j=0; j<end; j++) { 
        assert(j<nnodes);
        nodes[j].data.reset();
      }
      int end2 = nedges;
      int chunk1 = nedges/nthreads;
      #pragma omp for schedule(static, chunk1) private(end2)
      for (int k=0; k<end2; ++k) {
        assert(k<nedges);
        edgeData[k].reset();
      }
    }
  }
  */

  void toucDistGraph() {
    int sum = 0;
    for (int j=0; j<nnodes; j++) { 
      sum += nodes[j].data.id;
    }

    for (int j=0; j<nedges; ++j) {
      sum += edgeData[j].level;
    }
  }

  void verify() {
    double sampleBC = 0.0;
    bool firstTime = true;
    for (int i=0; i<nnodes; ++i) {
      const ND & n = nodes[i].data;
      if (firstTime) {
        sampleBC = n.bc;
        std::cerr << "BC: " << sampleBC << std::endl;
        firstTime = false;
      } else {
        if (!((n.bc - sampleBC) <= 0.0001)) 
          std::cerr << "Verification failed! " << (n.bc - sampleBC) << std::endl;
        assert ((n.bc - sampleBC) <= 0.0001);
      }
    }
    std::cerr << "Verification ok!" << std::endl;
  }

  void printBCs() {
    //for (int i=0; i<nnodes; ++i) {
    int n = std::min(nnodes,10);
    for (int i=0; i<n; ++i) {
      std::cerr << i << ": " << std::setiosflags(std::ios::fixed) << std::setprecision(6) << nodes[i].data.bc << std::endl;
    }
  }

  void printAllBCs(int numThreads, char *filename) {
    std::cerr << "Writting out bc values...\n";
    std::stringstream outfname;
    outfname << filename << "_" << numThreads << ".txt";
    std::string fname = outfname.str();
    std::ofstream outfile(fname.c_str());
    for (int i=0; i<nnodes; ++i) {
      outfile << i << " " << std::setiosflags(std::ios::fixed) << std::setprecision(6) << nodes[i].data.bc << std::endl;
    }
    outfile.close();
  }

  void printGraph() {
    std::cerr << "Nodes: " << std::endl;
    for (int i=0; i<nnodes; ++i) {
      std::cerr << nodes[i].data.toString() << std::endl;
    }
    /*for (int i=0; i<nedges; ++i) {
      std::cerr << i << " " ;
      std::cerr << edgeData[i].toString() << std::endl;
    }*/
    /*for (int i=0; i<ninIdx; ++i) {
      std::cerr << i << " " << inIdx[i] << std::endl;
    }*/

  }

  void checkSteadyState2() {
    std::cerr << "Doing second set of checks on graph...\n";
    for (int i=0; i<nnodes; ++i) {
      const ND & nodeD = nodes[i].data;
      if (nodeD.nsuccs != 0) 
        std::cerr << "Problem with nsuccs " << nodeD.nsuccs << std::endl;
      assert (nodeD.nsuccs == 0 || nodeD.nsuccs == -1);// : "Problem with nsuccs " + nodeD;
      //assert (nodeD->deltaDone());// : "Problem with DD " + nodeD;    
    }
  }

  void checkGraph(const ND * start) const {
    std::cerr << "Doing checks on graph...\n";
    for (int i=0; i<nnodes; ++i) {
      checkNode(&(nodes[i].data), start);
    }
  }
  
  void checkNode(const ND * n, const ND * source) const {
    const int idx = n->id;
    int start = outIdx[idx];
    int end = outIdx[idx + 1];
    int nSuccs = 0;
    for (int i = start; i < end; i++) {
      const ED & e = edgeData[i];
      const ND *nbr = e.dst;
      if (nbr->distance == n->distance + 1) {
        nSuccs++;
      }
    }
    if (nSuccs != n->nsuccs) {
      std::list<const ND*> theSuccs;
      for (int i = start; i < end; i++) {
        const ED & e = edgeData[i];
        const ND *nbr = e.dst;
        if (nbr->distance == n->distance + 1) {
          theSuccs.push_back(nbr);
        }
      }      
      std::cerr << "Successors problem : " << n->id << " " << n->nsuccs << " vs " << nSuccs << " " << std::endl;
      std::cerr << "Recorded successors are: ";
      for (std::list<const ND*>::const_iterator it = theSuccs.begin(); it != theSuccs.end(); ++it)
        std::cerr << (*it)->toString() << " ";
      std::cerr << std::endl;
      assert (false); 
    }
    
    start = inIdx[idx];
    end = inIdx[idx + 1];
    double sigma = 0;
    for (int i = start; i < end; i++) {
      const ED & e = edgeData[ins[i]];
      const ND *nbr = e.src;
      if (nbr->distance + 1 == n->distance) {
        if (!n->predsContain(nbr)) {  
          std::cerr << "Preds Problem: " << n->id << " " << nbr->toString() << std::endl;
          assert (false);
        }
        sigma += nbr->sigma;
      }
    }
    if (n != source) {
      if (sigma != n->sigma) {
        std::cerr << "Problem with sigma " << n->toString() << " vs. " << sigma << std::endl;
      } 
      assert (sigma == n->sigma);// : "Sigma problem " + n + " " + sigma;
    }
  }
};
#endif
