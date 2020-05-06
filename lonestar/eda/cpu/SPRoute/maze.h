#ifndef _MAZE_H_
#define _MAZE_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <functional>
#include "bitmap_image.hpp"
#include "DataType.h"
#include "flute.h"
#include "DataProc.h"
#include "route.h"
#include "RipUp.h"

#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/PerThreadContainer.h"
#include "galois/Reduction.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/substrate/NumaMem.h"
#include "galois/AtomicHelpers.h"
#include "galois/runtime/Profile.h"

#include "galois/LargeArray.h"
#include "llvm/Support/CommandLine.h"

// using namespace std;

#define PARENT(i) (i - 1) / 2
//#define PARENT(i) ((i-1)>>1)
#define LEFT(i) 2 * i + 1
#define RIGHT(i) 2 * i + 2

#define NET_PARALLEL 0

int PRINT;
int PRINT_HEAT;
typedef struct {
  bool operator()(float* left, float* right) { return (*left) > (*right); }
} maze_greater;

class pq_grid {
public:
  float* d1_p;
  float d1_push;
  pq_grid() {
    d1_p    = NULL;
    d1_push = 0;
  };
  pq_grid(float* d1_p, float d1_push) {
    this->d1_p    = d1_p;
    this->d1_push = d1_push;
  }
};

class lateUpdateReq {
public:
  std::atomic<float>* d1_p;
  float d1_push;
  short parentX;
  short parentY;
  bool HV;
  lateUpdateReq() {
    d1_p    = NULL;
    d1_push = 0;
    parentX = 0;
    parentY = 0;
    HV      = false; // 1 == true, 3 == false
  };
  lateUpdateReq(std::atomic<float>* d1_p, float d1_push, short parentX,
                short parentY, bool HV) {
    this->d1_p    = d1_p;
    this->d1_push = d1_push;
    this->parentX = parentX;
    this->parentY = parentY;
    this->HV      = HV;
  }
};

typedef struct {
  bool operator()(const pq_grid& left, const pq_grid& right) const {
    return left.d1_push < right.d1_push;
  }
} pq_less;

/*typedef galois::PerThreadDeque< float* > PerThread_PQ;
typedef galois::gstl::Deque< float* > local_pq;*/ //FIFO TRIAL

typedef galois::PerThreadMinHeap<pq_grid, pq_less> PerThread_PQ;
typedef galois::gstl::PQ<pq_grid, pq_less> local_pq;

typedef galois::PerThreadVector<int> PerThread_Vec;
typedef galois::gstl::Vector<int> local_vec;

typedef struct {
  int x; // x position
  int y; // y position
} Pos;

#define FIFO_CHUNK_SIZE 4
#define OBIM_delta 20

auto RequestIndexer = [](const pq_grid& top) {
  return (unsigned int)(top.d1_push) /
         max(OBIM_delta, (int)(costheight / (2 * slope)));
};

auto RequestIndexerLate = [](const lateUpdateReq& top) {
  return (unsigned int)(top.d1_push) / OBIM_delta;
};

/*auto RequestIndexerConcurrent = [&](const concurrent_pq_grid& top) {
    return (unsigned int)(top.d1_push) / OBIM_delta;
};*/

namespace gwl = galois::worklists;
using PSChunk = gwl::PerThreadChunkFIFO<FIFO_CHUNK_SIZE>;
using OBIM    = gwl::OrderedByIntegerMetric<decltype(RequestIndexer), PSChunk>;
using OBIM_late =
    gwl::OrderedByIntegerMetric<decltype(RequestIndexerLate), PSChunk>;
// using OBIM_concurrent =
// gwl::OrderedByIntegerMetric<decltype(RequestIndexerConcurrent), PSChunk>;

struct THREAD_LOCAL_STORAGE {
  using LAptr = galois::substrate::LAptr;
  LAptr pop_heap2_LA;
  bool* pop_heap2;

  LAptr d1_p_LA, d1_alloc_LA;
  float** d1_p;
  float* d1_alloc;

  LAptr HV_p_LA, HV_alloc_LA, hyperV_p_LA, hyperV_alloc_LA, hyperH_p_LA,
      hyperH_alloc_LA;
  bool **HV_p, **hyperV_p, **hyperH_p;
  bool *HV_alloc, *hyperV_alloc, *hyperH_alloc;

  LAptr parentX1_p_LA, parentX1_alloc_LA, parentY1_p_LA, parentY1_alloc_LA,
      parentX3_p_LA, parentX3_alloc_LA, parentY3_p_LA, parentY3_alloc_LA;
  short **parentX1_p, **parentY1_p, **parentX3_p, **parentY3_p;
  short *parentX1_alloc, *parentY1_alloc, *parentX3_alloc, *parentY3_alloc;

  LAptr corrEdge_p_LA, corrEdge_alloc_LA;
  int** corrEdge_p;
  int* corrEdge_alloc;

  LAptr inRegion_p_LA, inRegion_alloc_LA;
  bool** inRegion_p;
  bool* inRegion_alloc;

  LAptr netEO_p_LA;
  OrderNetEdge* netEO_p;

  // maze_pq pq1;
  // std::vector<float*> v2;
  THREAD_LOCAL_STORAGE() {
    using namespace galois::substrate;

    if (NET_PARALLEL) {
      pop_heap2_LA = largeMallocLocal(yGrid * xGrid * sizeof(bool));
      pop_heap2    = reinterpret_cast<bool*>(pop_heap2_LA.get());

      d1_alloc_LA = largeMallocLocal(yGrid * xGrid * sizeof(float));
      d1_alloc    = reinterpret_cast<float*>(d1_alloc_LA.get());
      d1_p_LA     = largeMallocLocal(yGrid * sizeof(float*));
      d1_p        = reinterpret_cast<float**>(d1_p_LA.get());

      HV_alloc_LA     = largeMallocLocal(yGrid * xGrid * sizeof(bool));
      HV_alloc        = reinterpret_cast<bool*>(HV_alloc_LA.get());
      hyperV_alloc_LA = largeMallocLocal(yGrid * xGrid * sizeof(bool));
      hyperV_alloc    = reinterpret_cast<bool*>(hyperV_alloc_LA.get());
      hyperH_alloc_LA = largeMallocLocal(yGrid * xGrid * sizeof(bool));
      hyperH_alloc    = reinterpret_cast<bool*>(hyperH_alloc_LA.get());

      HV_p_LA     = largeMallocLocal(yGrid * sizeof(bool*));
      HV_p        = reinterpret_cast<bool**>(HV_p_LA.get());
      hyperV_p_LA = largeMallocLocal(yGrid * sizeof(bool*));
      hyperV_p    = reinterpret_cast<bool**>(hyperV_p_LA.get());
      hyperH_p_LA = largeMallocLocal(yGrid * sizeof(bool*));
      hyperH_p    = reinterpret_cast<bool**>(hyperH_p_LA.get());

      parentX1_alloc_LA = largeMallocLocal(yGrid * xGrid * sizeof(short));
      parentX1_alloc    = reinterpret_cast<short*>(parentX1_alloc_LA.get());
      parentX3_alloc_LA = largeMallocLocal(yGrid * xGrid * sizeof(short));
      parentX3_alloc    = reinterpret_cast<short*>(parentX3_alloc_LA.get());
      parentY1_alloc_LA = largeMallocLocal(yGrid * xGrid * sizeof(short));
      parentY1_alloc    = reinterpret_cast<short*>(parentY1_alloc_LA.get());
      parentY3_alloc_LA = largeMallocLocal(yGrid * xGrid * sizeof(short));
      parentY3_alloc    = reinterpret_cast<short*>(parentY1_alloc_LA.get());

      parentX1_p_LA = largeMallocLocal(yGrid * sizeof(short*));
      parentX1_p    = reinterpret_cast<short**>(parentX1_p_LA.get());
      parentX3_p_LA = largeMallocLocal(yGrid * sizeof(short*));
      parentX3_p    = reinterpret_cast<short**>(parentX3_p_LA.get());
      parentY1_p_LA = largeMallocLocal(yGrid * sizeof(short*));
      parentY1_p    = reinterpret_cast<short**>(parentY1_p_LA.get());
      parentY3_p_LA = largeMallocLocal(yGrid * sizeof(short*));
      parentY3_p    = reinterpret_cast<short**>(parentY3_p_LA.get());

      corrEdge_alloc_LA = largeMallocLocal(yGrid * xGrid * sizeof(int));
      corrEdge_alloc    = reinterpret_cast<int*>(corrEdge_alloc_LA.get());
      corrEdge_p_LA     = largeMallocLocal(yGrid * sizeof(int*));
      corrEdge_p        = reinterpret_cast<int**>(corrEdge_p_LA.get());

      inRegion_alloc_LA = largeMallocLocal(yGrid * xGrid * sizeof(bool));
      inRegion_alloc    = reinterpret_cast<bool*>(inRegion_alloc_LA.get());
      inRegion_p_LA     = largeMallocLocal(yGrid * sizeof(bool*));
      inRegion_p        = reinterpret_cast<bool**>(inRegion_p_LA.get());

      netEO_p_LA = largeMallocLocal(MAXNETDEG * 2 * sizeof(OrderNetEdge));
      netEO_p    = reinterpret_cast<OrderNetEdge*>(netEO_p_LA.get());
    } else {
      pop_heap2 = (bool*)calloc(yGrid * xGrid, sizeof(bool));

      d1_alloc = (float*)calloc(yGrid * xGrid, sizeof(float));
      d1_p     = (float**)calloc(yGrid, sizeof(float*));

      HV_alloc     = (bool*)calloc(yGrid * xGrid, sizeof(bool));
      hyperV_alloc = (bool*)calloc(yGrid * xGrid, sizeof(bool));
      hyperH_alloc = (bool*)calloc(yGrid * xGrid, sizeof(bool));
      HV_p         = (bool**)calloc(yGrid, sizeof(bool*));
      hyperV_p     = (bool**)calloc(yGrid, sizeof(bool*));
      hyperH_p     = (bool**)calloc(yGrid, sizeof(bool*));

      parentX1_alloc = (short*)calloc(yGrid * xGrid, sizeof(short));
      parentX3_alloc = (short*)calloc(yGrid * xGrid, sizeof(short));
      parentY1_alloc = (short*)calloc(yGrid * xGrid, sizeof(short));
      parentY3_alloc = (short*)calloc(yGrid * xGrid, sizeof(short));
      parentX1_p     = (short**)calloc(yGrid, sizeof(short*));
      parentX3_p     = (short**)calloc(yGrid, sizeof(short*));
      parentY1_p     = (short**)calloc(yGrid, sizeof(short*));
      parentY3_p     = (short**)calloc(yGrid, sizeof(short*));

      corrEdge_alloc = (int*)calloc(yGrid * xGrid, sizeof(int));
      corrEdge_p     = (int**)calloc(yGrid, sizeof(int*));

      inRegion_alloc = (bool*)calloc(yGrid * xGrid, sizeof(bool));
      inRegion_p     = (bool**)calloc(yGrid, sizeof(bool*));

      netEO_p = (OrderNetEdge*)calloc(MAXNETDEG * 2, sizeof(OrderNetEdge));
    }
    // printf("allocation success\n");
    for (int i = 0; i < yGrid; i++) {
      d1_p[i] = &(d1_alloc[i * xGrid]);

      HV_p[i]     = &(HV_alloc[i * xGrid]);
      hyperV_p[i] = &(hyperV_alloc[i * xGrid]);
      hyperH_p[i] = &(hyperH_alloc[i * xGrid]);

      corrEdge_p[i] = &(corrEdge_alloc[i * xGrid]);

      inRegion_p[i] = &(inRegion_alloc[i * xGrid]);
    }

    for (int i = 0; i < yGrid; i++) {
      parentX1_p[i] = &(parentX1_alloc[i * xGrid]);
      parentX3_p[i] = &(parentX3_alloc[i * xGrid]);
      parentY1_p[i] = &(parentY1_alloc[i * xGrid]);
      parentY3_p[i] = &(parentY3_alloc[i * xGrid]);
    }
  }
  void reset_heap() { memset(pop_heap2, 0, yGrid * xGrid * sizeof(bool)); }

  ~THREAD_LOCAL_STORAGE() {
    free(pop_heap2);

    free(d1_p);
    free(d1_alloc);

    free(HV_p);
    free(hyperV_p);
    free(hyperH_p);
    free(HV_alloc);
    free(hyperV_alloc);
    free(hyperH_alloc);

    free(parentX1_p);
    free(parentY1_p);
    free(parentX3_p);
    free(parentY3_p);

    free(parentX1_alloc);
    free(parentY1_alloc);
    free(parentX3_alloc);
    free(parentY3_alloc);

    free(corrEdge_alloc);
    free(corrEdge_p);

    free(inRegion_alloc);
    free(inRegion_p);

    free(netEO_p);
  }
};

void convertToMazerouteNet(int netID) {
  short *gridsX, *gridsY;
  int i, edgeID, edgelength;
  int n1, n2, x1, y1, x2, y2;
  int cnt, Zpoint;
  TreeEdge* treeedge;
  TreeNode* treenodes;

  treenodes = sttrees[netID].nodes;
  for (edgeID = 0; edgeID < 2 * sttrees[netID].deg - 3; edgeID++) {
    treeedge               = &(sttrees[netID].edges[edgeID]);
    edgelength             = treeedge->len;
    n1                     = treeedge->n1;
    n2                     = treeedge->n2;
    x1                     = treenodes[n1].x;
    y1                     = treenodes[n1].y;
    x2                     = treenodes[n2].x;
    y2                     = treenodes[n2].y;
    treeedge->route.gridsX = (short*)calloc((edgelength + 1), sizeof(short));
    treeedge->route.gridsY = (short*)calloc((edgelength + 1), sizeof(short));
    gridsX                 = treeedge->route.gridsX;
    gridsY                 = treeedge->route.gridsY;
    treeedge->len          = ADIFF(x1, x2) + ADIFF(y1, y2);

    cnt = 0;
    if (treeedge->route.type == NOROUTE) {
      gridsX[0]                = x1;
      gridsY[0]                = y1;
      treeedge->route.type     = MAZEROUTE;
      treeedge->route.routelen = 0;
      treeedge->len            = 0;
      cnt++;
    } else if (treeedge->route.type == LROUTE) {
      if (treeedge->route.xFirst) // horizontal first
      {
        for (i = x1; i <= x2; i++) {
          gridsX[cnt] = i;
          gridsY[cnt] = y1;
          cnt++;
        }
        if (y1 <= y2) {
          for (i = y1 + 1; i <= y2; i++) {
            gridsX[cnt] = x2;
            gridsY[cnt] = i;
            cnt++;
          }
        } else {
          for (i = y1 - 1; i >= y2; i--) {
            gridsX[cnt] = x2;
            gridsY[cnt] = i;
            cnt++;
          }
        }
      } else // vertical first
      {
        if (y1 <= y2) {
          for (i = y1; i <= y2; i++) {
            gridsX[cnt] = x1;
            gridsY[cnt] = i;
            cnt++;
          }
        } else {
          for (i = y1; i >= y2; i--) {
            gridsX[cnt] = x1;
            gridsY[cnt] = i;
            cnt++;
          }
        }
        for (i = x1 + 1; i <= x2; i++) {
          gridsX[cnt] = i;
          gridsY[cnt] = y2;
          cnt++;
        }
      }
    } else if (treeedge->route.type == ZROUTE) {
      Zpoint = treeedge->route.Zpoint;
      if (treeedge->route.HVH) // HVH
      {
        for (i = x1; i < Zpoint; i++) {
          gridsX[cnt] = i;
          gridsY[cnt] = y1;
          cnt++;
        }
        if (y1 <= y2) {
          for (i = y1; i <= y2; i++) {
            gridsX[cnt] = Zpoint;
            gridsY[cnt] = i;
            cnt++;
          }
        } else {
          for (i = y1; i >= y2; i--) {
            gridsX[cnt] = Zpoint;
            gridsY[cnt] = i;
            cnt++;
          }
        }
        for (i = Zpoint + 1; i <= x2; i++) {
          gridsX[cnt] = i;
          gridsY[cnt] = y2;
          cnt++;
        }
      } else // VHV
      {
        if (y1 <= y2) {
          for (i = y1; i < Zpoint; i++) {
            gridsX[cnt] = x1;
            gridsY[cnt] = i;
            cnt++;
          }
          for (i = x1; i <= x2; i++) {
            gridsX[cnt] = i;
            gridsY[cnt] = Zpoint;
            cnt++;
          }
          for (i = Zpoint + 1; i <= y2; i++) {
            gridsX[cnt] = x2;
            gridsY[cnt] = i;
            cnt++;
          }
        } else {
          for (i = y1; i > Zpoint; i--) {
            gridsX[cnt] = x1;
            gridsY[cnt] = i;
            cnt++;
          }
          for (i = x1; i <= x2; i++) {
            gridsX[cnt] = i;
            gridsY[cnt] = Zpoint;
            cnt++;
          }
          for (i = Zpoint - 1; i >= y2; i--) {
            gridsX[cnt] = x2;
            gridsY[cnt] = i;
            cnt++;
          }
        }
      }
    }

    treeedge->route.type     = MAZEROUTE;
    treeedge->route.routelen = edgelength;

  } // loop for all the edges
}

void convertToMazeroute() {
  int i, j, netID;

  for (netID = 0; netID < numValidNets; netID++) {
    convertToMazerouteNet(netID);
  }

  for (i = 0; i < yGrid; i++) {
    for (j = 0; j < xGrid - 1; j++) {
      int grid            = i * (xGrid - 1) + j;
      h_edges[grid].usage = h_edges[grid].est_usage;
    }
  }
  //    fprintf(fpv, "\nVertical Congestion\n");
  for (i = 0; i < yGrid - 1; i++) {
    for (j = 0; j < xGrid; j++) {
      int grid            = i * xGrid + j;
      v_edges[grid].usage = v_edges[grid].est_usage;
    }
  }
}

// non recursive version of heapify
void heapify(float** array, int heapSize, int i) {
  int l, r, smallest;
  float* tmp;
  Bool STOP = FALSE;

  tmp = array[i];
  do {

    l = LEFT(i);
    r = RIGHT(i);

    if (l < heapSize && *(array[l]) < *tmp) {
      smallest = l;
      if (r < heapSize && *(array[r]) < *(array[l]))
        smallest = r;
    } else {
      smallest = i;
      if (r < heapSize && *(array[r]) < *tmp)
        smallest = r;
    }
    if (smallest != i) {
      array[i] = array[smallest];
      i        = smallest;
    } else {
      array[i] = tmp;
      STOP     = TRUE;
    }
  } while (!STOP);
}

// build heap for an list of grid
/*void buildHeap(float **array, int arrayLen)
{
    int i;

    for (i=arrayLen/2-1; i>=0; i--)
        heapify(array, arrayLen, i);
}*/

void updateHeap(float** array, int i) {
  int parent;
  float* tmpi;

  tmpi = array[i];
  while (i > 0 && *(array[PARENT(i)]) > *tmpi) {
    parent   = PARENT(i);
    array[i] = array[parent];
    i        = parent;
  }
  array[i] = tmpi;
}

// extract the entry with minimum distance from Priority queue
void extractMin(float** array, int arrayLen) {

  //    if(arrayLen<1)
  //        printf("Error: heap underflow\n");
  array[0] = array[arrayLen - 1];
  heapify(array, arrayLen - 1, 0);
}

/*
 * num_iteration : the total number of iterations for maze route to run
 * round : the number of maze route stages runned
 */

void updateCongestionHistory(int upType) {
  int i, j, grid, maxlimit;
  float overflow;

  maxlimit = 0;

  printf("updateType %d\n", upType);

  if (upType == 1) {
    for (i = 0; i < yGrid; i++) {
      for (j = 0; j < xGrid - 1; j++) {
        grid     = i * (xGrid - 1) + j;
        overflow = h_edges[grid].usage - h_edges[grid].cap;

        if (overflow > 0) {
          h_edges[grid].last_usage += overflow;
          h_edges[grid].congCNT++;
        } else {
          if (!stopDEC) {
            h_edges[grid].last_usage = h_edges[grid].last_usage * 0.9;
          }
        }
        maxlimit = max(maxlimit, h_edges[grid].last_usage);
      }
    }

    for (i = 0; i < yGrid - 1; i++) {
      for (j = 0; j < xGrid; j++) {
        grid     = i * xGrid + j;
        overflow = v_edges[grid].usage - v_edges[grid].cap;

        if (overflow > 0) {
          v_edges[grid].last_usage += overflow;
          v_edges[grid].congCNT++;
        } else {
          if (!stopDEC) {
            v_edges[grid].last_usage = v_edges[grid].last_usage * 0.9;
          }
        }
        maxlimit = max(maxlimit, v_edges[grid].last_usage);
      }
    }
  } else if (upType == 2) {
    if (max_adj < ahTH) {
      stopDEC = TRUE;
    } else {
      stopDEC = FALSE;
    }
    for (i = 0; i < yGrid; i++) {
      for (j = 0; j < xGrid - 1; j++) {
        grid     = i * (xGrid - 1) + j;
        overflow = h_edges[grid].usage - h_edges[grid].cap;

        if (overflow > 0) {
          h_edges[grid].congCNT++;
          h_edges[grid].last_usage += overflow;
        } else {
          if (!stopDEC) {
            h_edges[grid].congCNT--;
            h_edges[grid].congCNT    = max(0, h_edges[grid].congCNT);
            h_edges[grid].last_usage = h_edges[grid].last_usage * 0.9;
          }
        }
        maxlimit = max(maxlimit, h_edges[grid].last_usage);
      }
    }

    for (i = 0; i < yGrid - 1; i++) {
      for (j = 0; j < xGrid; j++) {
        grid     = i * xGrid + j;
        overflow = v_edges[grid].usage - v_edges[grid].cap;

        if (overflow > 0) {
          v_edges[grid].congCNT++;
          v_edges[grid].last_usage += overflow;
        } else {
          if (!stopDEC) {
            v_edges[grid].congCNT--;
            v_edges[grid].congCNT    = max(0, v_edges[grid].congCNT);
            v_edges[grid].last_usage = v_edges[grid].last_usage * 0.9;
          }
        }
        maxlimit = max(maxlimit, v_edges[grid].last_usage);
      }
    }

  } else if (upType == 3) {
    for (i = 0; i < yGrid; i++) {
      for (j = 0; j < xGrid - 1; j++) {
        grid     = i * (xGrid - 1) + j;
        overflow = h_edges[grid].usage - h_edges[grid].cap;

        if (overflow > 0) {
          h_edges[grid].congCNT++;
          h_edges[grid].last_usage += overflow;
        } else {
          if (!stopDEC) {
            h_edges[grid].congCNT--;
            h_edges[grid].congCNT = max(0, h_edges[grid].congCNT);
            h_edges[grid].last_usage += overflow;
            h_edges[grid].last_usage = max(h_edges[grid].last_usage, 0);
          }
        }
        maxlimit = max(maxlimit, h_edges[grid].last_usage);
      }
    }

    for (i = 0; i < yGrid - 1; i++) {
      for (j = 0; j < xGrid; j++) {
        grid     = i * xGrid + j;
        overflow = v_edges[grid].usage - v_edges[grid].cap;

        if (overflow > 0) {
          v_edges[grid].congCNT++;
          v_edges[grid].last_usage += overflow;
        } else {
          if (!stopDEC) {
            v_edges[grid].congCNT--;
            v_edges[grid].last_usage += overflow;
            v_edges[grid].last_usage = max(v_edges[grid].last_usage, 0);
          }
        }
        maxlimit = max(maxlimit, v_edges[grid].last_usage);
      }
    }

  } else if (upType == 4) {
    for (i = 0; i < yGrid; i++) {
      for (j = 0; j < xGrid - 1; j++) {
        grid     = i * (xGrid - 1) + j;
        overflow = h_edges[grid].usage - h_edges[grid].cap;

        if (overflow > 0) {
          h_edges[grid].congCNT++;
          h_edges[grid].last_usage += overflow;
        } else {
          if (!stopDEC) {
            h_edges[grid].congCNT--;
            h_edges[grid].congCNT    = max(0, h_edges[grid].congCNT);
            h_edges[grid].last_usage = h_edges[grid].last_usage * 0.9;
          }
        }
        maxlimit = max(maxlimit, h_edges[grid].last_usage);
      }
    }

    for (i = 0; i < yGrid - 1; i++) {
      for (j = 0; j < xGrid; j++) {
        grid     = i * xGrid + j;
        overflow = v_edges[grid].usage - v_edges[grid].cap;

        if (overflow > 0) {
          v_edges[grid].congCNT++;
          v_edges[grid].last_usage += overflow;
        } else {
          if (!stopDEC) {
            v_edges[grid].congCNT--;
            v_edges[grid].congCNT    = max(0, v_edges[grid].congCNT);
            v_edges[grid].last_usage = v_edges[grid].last_usage * 0.9;
          }
        }
        maxlimit = max(maxlimit, v_edges[grid].last_usage);
      }
    }
    //	if (maxlimit < 20) {
    //		stopDEC = TRUE;
    //	}
  }

  max_adj = maxlimit;

  printf("max value %d stop %d\n", maxlimit, stopDEC);
}

// ripup a tree edge according to its ripup type and Z-route it
// put all the nodes in the subtree t1 and t2 into heap1 and heap2
// netID   - the ID for the net
// edgeID  - the ID for the tree edge to route
// d1      - the distance of any grid from the source subtree t1
// d2      - the distance of any grid from the destination subtree t2
// heap1   - the heap storing the addresses for d1[][]
// heap2   - the heap storing the addresses for d2[][]
void setupHeap(int netID, int edgeID, local_pq& pq1, local_vec& v2,
               int regionX1, int regionX2, int regionY1, int regionY2,
               float** d1, int** corrEdge, bool** inRegion) {
  int i, j, d, numNodes, n1, n2, x1, y1, x2, y2;
  int nbr, nbrX, nbrY, cur, edge;
  int x_grid, y_grid;
  int queuehead, queuetail, *queue;
  Bool* visited;
  TreeEdge* treeedges;
  TreeNode* treenodes;
  Route* route;

  for (i = regionY1; i <= regionY2; i++) {
    for (j = regionX1; j <= regionX2; j++)
      inRegion[i][j] = TRUE;
  }

  treeedges = sttrees[netID].edges;
  treenodes = sttrees[netID].nodes;
  d         = sttrees[netID].deg;

  n1 = treeedges[edgeID].n1;
  n2 = treeedges[edgeID].n2;
  x1 = treenodes[n1].x;
  y1 = treenodes[n1].y;
  x2 = treenodes[n2].x;
  y2 = treenodes[n2].y;

  // if(netID == 14628)
  //    printf("net: %d edge: %d src: %d %d dst: %d %d d: %d\n", netID, edgeID,
  //    y1, x1, y2, x2, d);
  pq1.clear();
  v2.clear(); // Michael
  if (d == 2) // 2-pin net
  {
    d1[y1][x1] = 0;
    pq1.push({&(d1[y1][x1]), 0});
    v2.push_back(y2 * xGrid + x2);
  } else // net with more than 2 pins
  {
    numNodes = 2 * d - 2;

    visited = (Bool*)calloc(numNodes, sizeof(Bool));
    for (i = 0; i < numNodes; i++)
      visited[i] = FALSE;

    queue = (int*)calloc(numNodes, sizeof(int));

    // find all the grids on tree edges in subtree t1 (connecting to n1) and put
    // them into heap1
    if (n1 < d) // n1 is a Pin node
    {
      // just need to put n1 itself into heap1
      d1[y1][x1] = 0;
      pq1.push({&(d1[y1][x1]), 0});
      visited[n1] = TRUE;
    } else // n1 is a Steiner node
    {
      queuehead = queuetail = 0;

      // add n1 into heap1
      d1[y1][x1] = 0;
      // if(netID == 252163 && edgeID == 51)
      //    printf("y: %d x: %d\n", y1, x1);
      pq1.push({&(d1[y1][x1]), 0});
      visited[n1] = TRUE;

      // add n1 into the queue
      queue[queuetail] = n1;
      queuetail++;

      // loop to find all the edges in subtree t1
      while (queuetail > queuehead) {
        // get cur node from the queuehead
        cur = queue[queuehead];
        queuehead++;
        visited[cur] = TRUE;
        if (cur >= d) // cur node is a Steiner node
        {
          for (i = 0; i < 3; i++) {
            nbr  = treenodes[cur].nbr[i];
            edge = treenodes[cur].edge[i];
            if (nbr != n2) // not n2
            {
              if (visited[nbr] == FALSE) {
                // put all the grids on the two adjacent tree edges into heap1
                if (treeedges[edge].route.routelen > 0) // not a degraded edge
                {
                  // put nbr into heap1 if in enlarged region
                  if (inRegion[treenodes[nbr].y][treenodes[nbr].x]) {
                    nbrX           = treenodes[nbr].x;
                    nbrY           = treenodes[nbr].y;
                    d1[nbrY][nbrX] = 0;
                    // if(netID == 252163 && edgeID == 51)
                    //    printf("y: %d x: %d\n", nbrY, nbrX);
                    pq1.push({&(d1[nbrY][nbrX]), 0});
                    corrEdge[nbrY][nbrX] = edge;
                  }

                  // the coordinates of two end nodes of the edge

                  route = &(treeedges[edge].route);
                  if (route->type == MAZEROUTE) {
                    for (j = 1; j < route->routelen;
                         j++) // don't put edge_n1 and edge_n2 into heap1
                    {
                      x_grid = route->gridsX[j];
                      y_grid = route->gridsY[j];

                      if (inRegion[y_grid][x_grid]) {
                        d1[y_grid][x_grid] = 0;
                        // if(netID == 252163 && edgeID == 51)
                        //    printf("y: %d x: %d\n", y_grid, x_grid);
                        pq1.push({&(d1[y_grid][x_grid]), 0});
                        corrEdge[y_grid][x_grid] = edge;
                      }
                    }
                  } // if MAZEROUTE
                  else {
                    printf("Setup Heap: not maze routing\n");
                  }
                } // if not a degraded edge (len>0)

                // add the neighbor of cur node into queue
                queue[queuetail] = nbr;
                queuetail++;
              } // if the node is not visited
            }   // if nbr!=n2
          }     // loop i (3 neigbors for cur node)
        }       // if cur node is a Steiner nodes
      }         // while queue is not empty
    }           // else n1 is not a Pin node

    // find all the grids on subtree t2 (connect to n2) and put them into heap2
    // find all the grids on tree edges in subtree t2 (connecting to n2) and put
    // them into heap2
    if (n2 < d) // n2 is a Pin node
    {
      // just need to put n2 itself into heap2
      v2.push_back(y2 * xGrid + x2);
      // if(netID == 14628)
      //    printf("y: %d x: %d \n", y2, x2);
      visited[n2] = TRUE;
    } else // n2 is a Steiner node
    {
      queuehead = queuetail = 0;

      // add n2 into heap2
      v2.push_back(y2 * xGrid + x2);
      // if(netID == 252163 && edgeID == 51)
      //    printf("dst y: %d x: %d \n", y2, x2);
      visited[n2] = TRUE;

      // add n2 into the queue
      queue[queuetail] = n2;
      queuetail++;

      // loop to find all the edges in subtree t2
      while (queuetail > queuehead) {
        // get cur node form queuehead
        cur          = queue[queuehead];
        visited[cur] = TRUE;
        queuehead++;

        if (cur >= d) // cur node is a Steiner node
        {
          for (i = 0; i < 3; i++) {
            nbr  = treenodes[cur].nbr[i];
            edge = treenodes[cur].edge[i];
            if (nbr != n1) // not n1
            {
              if (visited[nbr] == FALSE) {
                // put all the grids on the two adjacent tree edges into heap2
                if (treeedges[edge].route.routelen > 0) // not a degraded edge
                {
                  // put nbr into heap2
                  if (inRegion[treenodes[nbr].y][treenodes[nbr].x]) {
                    nbrX = treenodes[nbr].x;
                    nbrY = treenodes[nbr].y;
                    v2.push_back(nbrY * xGrid + nbrX);
                    // if(netID == 252163 && edgeID == 51)
                    //    printf("dst y: %d x: %d\n", nbrY, nbrX);
                    corrEdge[nbrY][nbrX] = edge;
                  }

                  // the coordinates of two end nodes of the edge

                  route = &(treeedges[edge].route);
                  if (route->type == MAZEROUTE) {
                    for (j = 1; j < route->routelen;
                         j++) // don't put edge_n1 and edge_n2 into heap2
                    {
                      x_grid = route->gridsX[j];
                      y_grid = route->gridsY[j];
                      if (inRegion[y_grid][x_grid]) {
                        v2.push_back(y_grid * xGrid + x_grid);
                        // if(netID == 252163 && edgeID == 51)
                        //    printf("dst y: %d x: %d\n", y_grid, x_grid);
                        corrEdge[y_grid][x_grid] = edge;
                      }
                    }
                  } // if MAZEROUTE
                  else {
                    printf("Setup Heap: not maze routing\n");
                  }
                } // if the edge is not degraded (len>0)

                // add the neighbor of cur node into queue
                queue[queuetail] = nbr;
                queuetail++;
              } // if the node is not visited
            }   // if nbr!=n1
          }     // loop i (3 neigbors for cur node)
        }       // if cur node is a Steiner nodes
      }         // while queue is not empty
    }           // else n2 is not a Pin node

    free(queue);
    free(visited);
  } // net with more than two pins

  for (i = regionY1; i <= regionY2; i++) {
    for (j = regionX1; j <= regionX2; j++)
      inRegion[i][j] = FALSE;
  }
}

int copyGrids(TreeNode* treenodes, int n1, TreeEdge* treeedges, int edge_n1n2,
              int* gridsX_n1n2, int* gridsY_n1n2) {
  int i, cnt;
  int n1x, n1y;

  n1x = treenodes[n1].x;
  n1y = treenodes[n1].y;

  cnt = 0;
  if (treeedges[edge_n1n2].n1 == n1) // n1 is the first node of (n1, n2)
  {
    if (treeedges[edge_n1n2].route.type == MAZEROUTE) {
      for (i = 0; i <= treeedges[edge_n1n2].route.routelen; i++) {
        gridsX_n1n2[cnt] = treeedges[edge_n1n2].route.gridsX[i];
        gridsY_n1n2[cnt] = treeedges[edge_n1n2].route.gridsY[i];
        cnt++;
      }
    }    // MAZEROUTE
    else // NOROUTE
    {
      gridsX_n1n2[cnt] = n1x;
      gridsY_n1n2[cnt] = n1y;
      cnt++;
    }
  }    // if n1 is the first node of (n1, n2)
  else // n2 is the first node of (n1, n2)
  {
    if (treeedges[edge_n1n2].route.type == MAZEROUTE) {
      for (i = treeedges[edge_n1n2].route.routelen; i >= 0; i--) {
        gridsX_n1n2[cnt] = treeedges[edge_n1n2].route.gridsX[i];
        gridsY_n1n2[cnt] = treeedges[edge_n1n2].route.gridsY[i];
        cnt++;
      }
    }    // MAZEROUTE
    else // NOROUTE
    {
      gridsX_n1n2[cnt] = n1x;
      gridsY_n1n2[cnt] = n1y;
      cnt++;
    } // MAZEROUTE
  }

  return (cnt);
}

void updateRouteType1(TreeNode* treenodes, int n1, int A1, int A2, int E1x,
                      int E1y, TreeEdge* treeedges, int edge_n1A1,
                      int edge_n1A2) {
  using namespace galois::substrate;
  int i, cnt, A1x, A1y, A2x, A2y;
  int cnt_n1A1, cnt_n1A2, E1_pos = 0;
  // int gridsXY[4 * (xGrid + yGrid)];

  int gridsX_n1A1[2 * (xGrid + yGrid)], gridsY_n1A1[2 * (xGrid + yGrid)],
      gridsX_n1A2[2 * (xGrid + yGrid)], gridsY_n1A2[2 * (xGrid + yGrid)];

  /*int* gridsX_n1A1 = gridsXY;
  int* gridsY_n1A1 = gridsXY + xGrid + yGrid;
  int* gridsX_n1A2 = gridsXY + 2 * (xGrid + yGrid);
  int* gridsY_n1A2 = gridsXY + 3 * (xGrid + yGrid);*/

  /*LAptr gridsX_n1A1_LA, gridsY_n1A1_LA, gridsX_n1A2_LA, gridsY_n1A2_LA;
  gridsX_n1A1_LA = largeMallocLocal((xGrid + yGrid) * sizeof(int));
  gridsY_n1A1_LA = largeMallocLocal((xGrid + yGrid) * sizeof(int));
  gridsX_n1A2_LA = largeMallocLocal((xGrid + yGrid) * sizeof(int));
  gridsY_n1A2_LA = largeMallocLocal((xGrid + yGrid) * sizeof(int));

  int* gridsX_n1A1 = reinterpret_cast<int*> (gridsX_n1A1_LA.get());
  int* gridsY_n1A1 = reinterpret_cast<int*> (gridsY_n1A1_LA.get());
  int* gridsX_n1A2 = reinterpret_cast<int*> (gridsX_n1A2_LA.get());
  int* gridsY_n1A2 = reinterpret_cast<int*> (gridsY_n1A2_LA.get());*/

  A1x = treenodes[A1].x;
  A1y = treenodes[A1].y;
  A2x = treenodes[A2].x;
  A2y = treenodes[A2].y;

  // copy all the grids on (n1, A1) and (n2, A2) to tmp arrays, and keep the
  // grids order A1->n1->A2 copy (n1, A1)
  cnt_n1A1 =
      copyGrids(treenodes, A1, treeedges, edge_n1A1, gridsX_n1A1, gridsY_n1A1);

  // copy (n1, A2)
  cnt_n1A2 =
      copyGrids(treenodes, n1, treeedges, edge_n1A2, gridsX_n1A2, gridsY_n1A2);

  // update route for (n1, A1) and (n1, A2)
  // find the index of E1 in (n1, A1)
  for (i = 0; i < cnt_n1A1; i++) {
    if (gridsX_n1A1[i] == E1x && gridsY_n1A1[i] == E1y) // reach the E1
    {
      E1_pos = i;
      break;
    }
  }

  // reallocate memory for route.gridsX and route.gridsY
  if (treeedges[edge_n1A1].route.type ==
      MAZEROUTE) // if originally allocated, free them first
  {
    free(treeedges[edge_n1A1].route.gridsX);
    free(treeedges[edge_n1A1].route.gridsY);
  }
  treeedges[edge_n1A1].route.gridsX =
      (short*)calloc((E1_pos + 1), sizeof(short));
  treeedges[edge_n1A1].route.gridsY =
      (short*)calloc((E1_pos + 1), sizeof(short));

  if (A1x <= E1x) {
    cnt = 0;
    for (i = 0; i <= E1_pos; i++) {
      treeedges[edge_n1A1].route.gridsX[cnt] = gridsX_n1A1[i];
      treeedges[edge_n1A1].route.gridsY[cnt] = gridsY_n1A1[i];
      cnt++;
    }
    treeedges[edge_n1A1].n1 = A1;
    treeedges[edge_n1A1].n2 = n1;
  } else {
    cnt = 0;
    for (i = E1_pos; i >= 0; i--) {
      treeedges[edge_n1A1].route.gridsX[cnt] = gridsX_n1A1[i];
      treeedges[edge_n1A1].route.gridsY[cnt] = gridsY_n1A1[i];
      cnt++;
    }
    treeedges[edge_n1A1].n1 = n1;
    treeedges[edge_n1A1].n2 = A1;
  }

  treeedges[edge_n1A1].route.type     = MAZEROUTE;
  treeedges[edge_n1A1].route.routelen = E1_pos;
  treeedges[edge_n1A1].len            = ADIFF(A1x, E1x) + ADIFF(A1y, E1y);

  // reallocate memory for route.gridsX and route.gridsY
  if (treeedges[edge_n1A2].route.type ==
      MAZEROUTE) // if originally allocated, free them first
  {
    free(treeedges[edge_n1A2].route.gridsX);
    free(treeedges[edge_n1A2].route.gridsY);
  }
  treeedges[edge_n1A2].route.gridsX =
      (short*)calloc((cnt_n1A1 + cnt_n1A2 - E1_pos - 1), sizeof(short));
  treeedges[edge_n1A2].route.gridsY =
      (short*)calloc((cnt_n1A1 + cnt_n1A2 - E1_pos - 1), sizeof(short));

  if (E1x <= A2x) {
    cnt = 0;
    for (i = E1_pos; i < cnt_n1A1; i++) {
      treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A1[i];
      treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A1[i];
      cnt++;
    }
    for (i = 1; i < cnt_n1A2; i++) // 0 is n1 again, so no repeat
    {
      treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A2[i];
      treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A2[i];
      cnt++;
    }
    treeedges[edge_n1A2].n1 = n1;
    treeedges[edge_n1A2].n2 = A2;
  } else {
    cnt = 0;
    for (i = cnt_n1A2 - 1; i >= 1; i--) // 0 is n1 again, so no repeat
    {
      treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A2[i];
      treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A2[i];
      cnt++;
    }
    for (i = cnt_n1A1 - 1; i >= E1_pos; i--) {
      treeedges[edge_n1A2].route.gridsX[cnt] = gridsX_n1A1[i];
      treeedges[edge_n1A2].route.gridsY[cnt] = gridsY_n1A1[i];
      cnt++;
    }
    treeedges[edge_n1A2].n1 = A2;
    treeedges[edge_n1A2].n2 = n1;
  }
  treeedges[edge_n1A2].route.type     = MAZEROUTE;
  treeedges[edge_n1A2].route.routelen = cnt - 1;
  treeedges[edge_n1A2].len            = ADIFF(A2x, E1x) + ADIFF(A2y, E1y);
}

void updateRouteType2(TreeNode* treenodes, int n1, int A1, int A2, int C1,
                      int C2, int E1x, int E1y, TreeEdge* treeedges,
                      int edge_n1A1, int edge_n1A2, int edge_C1C2) {
  int i, cnt, A1x, A1y, A2x, A2y, C1x, C1y, C2x, C2y;
  int edge_n1C1, edge_n1C2, edge_A1A2;
  int cnt_n1A1, cnt_n1A2, cnt_C1C2, E1_pos = 0;
  int len_A1A2, len_n1C1, len_n1C2;
  int gridsX_n1A1[2 * (xGrid + yGrid)], gridsY_n1A1[2 * (xGrid + yGrid)];
  int gridsX_n1A2[2 * (xGrid + yGrid)], gridsY_n1A2[2 * (xGrid + yGrid)];
  int gridsX_C1C2[2 * (xGrid + yGrid)], gridsY_C1C2[2 * (xGrid + yGrid)];

  A1x = treenodes[A1].x;
  A1y = treenodes[A1].y;
  A2x = treenodes[A2].x;
  A2y = treenodes[A2].y;
  C1x = treenodes[C1].x;
  C1y = treenodes[C1].y;
  C2x = treenodes[C2].x;
  C2y = treenodes[C2].y;

  edge_n1C1 = edge_n1A1;
  edge_n1C2 = edge_n1A2;
  edge_A1A2 = edge_C1C2;

  // combine (n1, A1) and (n1, A2) into (A1, A2), A1 is the first node and A2 is
  // the second grids order A1->n1->A2 copy (A1, n1)
  cnt_n1A1 =
      copyGrids(treenodes, A1, treeedges, edge_n1A1, gridsX_n1A1, gridsY_n1A1);

  // copy (n1, A2)
  cnt_n1A2 =
      copyGrids(treenodes, n1, treeedges, edge_n1A2, gridsX_n1A2, gridsY_n1A2);

  // copy all the grids on (C1, C2) to gridsX_C1C2[] and gridsY_C1C2[]
  cnt_C1C2 =
      copyGrids(treenodes, C1, treeedges, edge_C1C2, gridsX_C1C2, gridsY_C1C2);

  // combine grids on original (A1, n1) and (n1, A2) to new (A1, A2)
  // allocate memory for gridsX[] and gridsY[] of edge_A1A2
  if (treeedges[edge_A1A2].route.type == MAZEROUTE) {
    free(treeedges[edge_A1A2].route.gridsX);
    free(treeedges[edge_A1A2].route.gridsY);
  }
  len_A1A2 = cnt_n1A1 + cnt_n1A2 - 1;

  treeedges[edge_A1A2].route.gridsX   = (short*)calloc(len_A1A2, sizeof(short));
  treeedges[edge_A1A2].route.gridsY   = (short*)calloc(len_A1A2, sizeof(short));
  treeedges[edge_A1A2].route.routelen = len_A1A2 - 1;
  treeedges[edge_A1A2].len            = ADIFF(A1x, A2x) + ADIFF(A1y, A2y);

  cnt = 0;
  for (i = 0; i < cnt_n1A1; i++) {
    treeedges[edge_A1A2].route.gridsX[cnt] = gridsX_n1A1[i];
    treeedges[edge_A1A2].route.gridsY[cnt] = gridsY_n1A1[i];
    cnt++;
  }
  for (i = 1; i < cnt_n1A2; i++) // do not repeat point n1
  {
    treeedges[edge_A1A2].route.gridsX[cnt] = gridsX_n1A2[i];
    treeedges[edge_A1A2].route.gridsY[cnt] = gridsY_n1A2[i];
    cnt++;
  }

  // find the index of E1 in (C1, C2)
  for (i = 0; i < cnt_C1C2; i++) {
    if (gridsX_C1C2[i] == E1x && gridsY_C1C2[i] == E1y) {
      E1_pos = i;
      break;
    }
  }

  // allocate memory for gridsX[] and gridsY[] of edge_n1C1 and edge_n1C2
  if (treeedges[edge_n1C1].route.type == MAZEROUTE) {
    free(treeedges[edge_n1C1].route.gridsX);
    free(treeedges[edge_n1C1].route.gridsY);
  }
  len_n1C1                            = E1_pos + 1;
  treeedges[edge_n1C1].route.gridsX   = (short*)calloc(len_n1C1, sizeof(short));
  treeedges[edge_n1C1].route.gridsY   = (short*)calloc(len_n1C1, sizeof(short));
  treeedges[edge_n1C1].route.routelen = len_n1C1 - 1;
  treeedges[edge_n1C1].len            = ADIFF(C1x, E1x) + ADIFF(C1y, E1y);

  if (treeedges[edge_n1C2].route.type == MAZEROUTE) {
    free(treeedges[edge_n1C2].route.gridsX);
    free(treeedges[edge_n1C2].route.gridsY);
  }
  len_n1C2                            = cnt_C1C2 - E1_pos;
  treeedges[edge_n1C2].route.gridsX   = (short*)calloc(len_n1C2, sizeof(short));
  treeedges[edge_n1C2].route.gridsY   = (short*)calloc(len_n1C2, sizeof(short));
  treeedges[edge_n1C2].route.routelen = len_n1C2 - 1;
  treeedges[edge_n1C2].len            = ADIFF(C2x, E1x) + ADIFF(C2y, E1y);

  // split original (C1, C2) to (C1, n1) and (n1, C2)
  cnt = 0;
  for (i = 0; i <= E1_pos; i++) {
    treeedges[edge_n1C1].route.gridsX[i] = gridsX_C1C2[i];
    treeedges[edge_n1C1].route.gridsY[i] = gridsY_C1C2[i];
    cnt++;
  }

  cnt = 0;
  for (i = E1_pos; i < cnt_C1C2; i++) {
    treeedges[edge_n1C2].route.gridsX[cnt] = gridsX_C1C2[i];
    treeedges[edge_n1C2].route.gridsY[cnt] = gridsY_C1C2[i];
    cnt++;
  }
}

void reInitTree(int netID) {
  int deg, numEdges, edgeID, d, j;
  TreeEdge* treeedge;
  Tree rsmt;
  int x[MAXNETDEG], y[MAXNETDEG];

  // printf("re init tree for net %d\n",netID);

  newRipupNet(netID);

  deg      = sttrees[netID].deg;
  numEdges = 2 * deg - 3;
  for (edgeID = 0; edgeID < numEdges; edgeID++) {
    treeedge = &(sttrees[netID].edges[edgeID]);
    if (treeedge->len > 0) {
      free(treeedge->route.gridsX);
      free(treeedge->route.gridsY);
      free(treeedge->route.gridsL);
    }
  }
  free(sttrees[netID].nodes);
  free(sttrees[netID].edges);

  // printf("old tree component freed\n");

  d = nets[netID]->deg;
  // printf("net deg %d\n",d);
  // fflush(stdout);
  for (j = 0; j < d; j++) {
    x[j] = nets[netID]->pinX[j];
    y[j] = nets[netID]->pinY[j];
  }
  // printf("before flute\n");
  // fflush(stdout);
  fluteCongest(netID, d, x, y, 2, 1.2, &rsmt);
  // printf("fluted worked\n");
  // fflush(stdout);
  if (d > 3) {
    edgeShiftNew(&rsmt);
    // printf("edge shifted\n");
  }
  // fflush(stdout);
  copyStTree(netID, rsmt);
  // printf("tree copied\n");
  // fflush(stdout);
  newrouteLInMaze(netID);
  // printf("L routing worked\n");
  // fflush(stdout);
  // newrouteZ(netID, 10);
  // printf("Z routign worked\n");
  // fflush(stdout);
  convertToMazerouteNet(netID);
  // printf("L to mzed converted\n");
  // fflush(stdout);
  // checkRoute2DTree(netID);
  // printf("tree double checked\n");
  // fflush(stdout);
}

void mazeRouteMSMD(int iter, int expand, float costHeight, int ripup_threshold,
                   int mazeedge_Threshold, Bool Ordering, int cost_type) {
  // LOCK = 0;
  float forange;

  // allocate memory for distance and parent and pop_heap
  h_costTable = (float*)calloc(40 * hCapacity, sizeof(float));
  v_costTable = (float*)calloc(40 * vCapacity, sizeof(float));

  forange = 40 * hCapacity;

  if (cost_type == 2) {
    for (int i = 0; i < forange; i++) {
      if (i < hCapacity - 1)
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i - 1) * LOGIS_COF) + 1) + 1;
      else
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i - 1) * LOGIS_COF) + 1) + 1 +
            costHeight / slope * (i - hCapacity);
    }
    forange = 40 * vCapacity;
    for (int i = 0; i < forange; i++) {
      if (i < vCapacity - 1)
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i - 1) * LOGIS_COF) + 1) + 1;
      else
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i - 1) * LOGIS_COF) + 1) + 1 +
            costHeight / slope * (i - vCapacity);
    }
  } else {

    for (int i = 0; i < forange; i++) {
      if (i < hCapacity)
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) + 1;
      else
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) + 1 +
            costHeight / slope * (i - hCapacity);
    }
    forange = 40 * vCapacity;
    for (int i = 0; i < forange; i++) {
      if (i < vCapacity)
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i) * LOGIS_COF) + 1) + 1;
      else
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i) * LOGIS_COF) + 1) + 1 +
            costHeight / slope * (i - vCapacity);
    }
  }
  // cout << " i = vCap:" << v_costTable[vCapacity-1] << " " <<
  // v_costTable[vCapacity] << " " << v_costTable[vCapacity+1] << endl;

  /*forange = yGrid*xGrid;
  for(int i=0; i<forange; i++)
  {
      pop_heap2[i] = FALSE;
  } //Michael*/

  if (Ordering) {
    StNetOrder();
    // printf("order?\n");
  }

  galois::substrate::PerThreadStorage<THREAD_LOCAL_STORAGE>
      thread_local_storage{};
  // for(nidRPC=0; nidRPC<numValidNets; nidRPC++)//parallelize
  PerThread_PQ perthread_pq;
  PerThread_Vec perthread_vec;
  PRINT = 0;
  galois::GAccumulator<int> total_ripups;
  galois::GReduceMax<int> max_ripups;
  total_ripups.reset();
  max_ripups.reset();

  // galois::runtime::profileVtune( [&] (void) {
  /*std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(net_shuffle.begin(), net_shuffle.end(), g);

  galois::do_all(galois::iterate(net_shuffle), */
  // galois::for_each(galois::iterate(0, numValidNets),
  //        [&] (const auto nidRPC, auto& ctx)
  galois::do_all(
      galois::iterate(0, numValidNets),
      [&](const auto nidRPC) {
        int grid, netID;

        // maze routing for multi-source, multi-destination
        bool hypered, enter;
        int i, j, deg, edgeID, n1, n2, n1x, n1y, n2x, n2y, ymin, ymax, xmin,
            xmax, curX, curY, crossX, crossY, tmpX, tmpY, tmpi, min_x, min_y,
            num_edges;
        int regionX1, regionX2, regionY1, regionY2;
        int ind1, tmpind, gridsX[XRANGE], gridsY[YRANGE], tmp_gridsX[XRANGE],
            tmp_gridsY[YRANGE];
        int endpt1, endpt2, A1, A2, B1, B2, C1, C2, D1, D2, cnt, cnt_n1n2;
        int edge_n1n2, edge_n1A1, edge_n1A2, edge_n1C1, edge_n1C2, edge_A1A2,
            edge_C1C2;
        int edge_n2B1, edge_n2B2, edge_n2D1, edge_n2D2, edge_B1B2, edge_D1D2;
        int E1x, E1y, E2x, E2y;
        int tmp_grid;
        int preX, preY, origENG, edgeREC;

        float tmp, tmp_cost;
        TreeEdge *treeedges, *treeedge;
        TreeNode* treenodes;

        bool* pop_heap2 = thread_local_storage.getLocal()->pop_heap2;

        float** d1    = thread_local_storage.getLocal()->d1_p;
        bool** HV     = thread_local_storage.getLocal()->HV_p;
        bool** hyperV = thread_local_storage.getLocal()->hyperV_p;
        bool** hyperH = thread_local_storage.getLocal()->hyperH_p;

        short** parentX1 = thread_local_storage.getLocal()->parentX1_p;
        short** parentX3 = thread_local_storage.getLocal()->parentX3_p;
        short** parentY1 = thread_local_storage.getLocal()->parentY1_p;
        short** parentY3 = thread_local_storage.getLocal()->parentY3_p;

        int** corrEdge = thread_local_storage.getLocal()->corrEdge_p;

        OrderNetEdge* netEO = thread_local_storage.getLocal()->netEO_p;

        bool** inRegion = thread_local_storage.getLocal()->inRegion_p;

        local_pq pq1 = perthread_pq.get();
        local_vec v2 = perthread_vec.get();

        /*for(i=0; i<yGrid*xGrid; i++)
        {
            pop_heap2[i] = FALSE;
        } */

        // memset(inRegion_alloc, 0, xGrid * yGrid * sizeof(bool));
        /*for(int i=0; i<yGrid; i++)
        {
            for(int j=0; j<xGrid; j++)
                inRegion[i][j] = FALSE;
        }*/
        // printf("hyperV[153][134]: %d %d %d\n", hyperV[153][134],
        // parentY1[153][134], parentX3[153][134]); printf("what is
        // happening?\n");

        if (Ordering) {
          netID = treeOrderCong[nidRPC].treeIndex;
        } else {
          netID = nidRPC;
        }

        deg = sttrees[netID].deg;

        origENG = expand;

        netedgeOrderDec(netID, netEO);

        treeedges = sttrees[netID].edges;
        treenodes = sttrees[netID].nodes;
        // loop for all the tree edges (2*deg-3)
        num_edges = 2 * deg - 3;

        for (edgeREC = 0; edgeREC < num_edges; edgeREC++) {
          edgeID   = netEO[edgeREC].edgeID;
          treeedge = &(treeedges[edgeID]);

          n1            = treeedge->n1;
          n2            = treeedge->n2;
          n1x           = treenodes[n1].x;
          n1y           = treenodes[n1].y;
          n2x           = treenodes[n2].x;
          n2y           = treenodes[n2].y;
          treeedge->len = ADIFF(n2x, n1x) + ADIFF(n2y, n1y);

          if (treeedge->len >
              mazeedge_Threshold) // only route the non-degraded edges (len>0)
          {
            // enter = newRipupCheck(treeedge, n1x, n1y, n2x, n2y,
            // ripup_threshold, netID, edgeID);
            enter =
                newRipupCheck_atomic(treeedge, ripup_threshold, netID, edgeID);

            // ripup the routing for the edge
            if (enter) {
              /*pre_length = treeedge->route.routelen;
              for(int i = 0; i < pre_length; i++)
              {
                  pre_gridsY[i] = treeedge->route.gridsY[i];
                  pre_gridsX[i] = treeedge->route.gridsX[i];
                  //printf("i %d x %d y %d\n", i, pre_gridsX[i], pre_gridsY[i]);
              }*/
              // if(netID == 252163 && edgeID == 51)
              //    printf("netID %d edgeID %d src %d %d dst %d %d\n", netID,
              //    edgeID, n1x, n1y, n2x, n2y);
              if (n1y <= n2y) {
                ymin = n1y;
                ymax = n2y;
              } else {
                ymin = n2y;
                ymax = n1y;
              }

              if (n1x <= n2x) {
                xmin = n1x;
                xmax = n2x;
              } else {
                xmin = n2x;
                xmax = n1x;
              }

              int enlarge =
                  min(origENG,
                      (iter / 6 + 3) *
                          treeedge->route
                              .routelen); // michael, this was global variable
              regionX1 = max(0, xmin - enlarge);
              regionX2 = min(xGrid - 1, xmax + enlarge);
              regionY1 = max(0, ymin - enlarge);
              regionY2 = min(yGrid - 1, ymax + enlarge);

              // initialize d1[][] and d2[][] as BIG_INT
              for (i = regionY1; i <= regionY2; i++) {
                for (j = regionX1; j <= regionX2; j++) {
                  d1[i][j] = BIG_INT;
                  /*d2[i][j] = BIG_INT;
                  hyperH[i][j] = FALSE;
                  hyperV[i][j] = FALSE;*/
                }
              }
              // memset(hyperH, 0, xGrid * yGrid * sizeof(bool));
              // memset(hyperV, 0, xGrid * yGrid * sizeof(bool));
              for (i = regionY1; i <= regionY2; i++) {
                for (j = regionX1; j <= regionX2; j++) {
                  hyperH[i][j] = FALSE;
                }
              }
              for (i = regionY1; i <= regionY2; i++) {
                for (j = regionX1; j <= regionX2; j++) {
                  hyperV[i][j] = FALSE;
                }
              }
              // TODO: use seperate loops

              // setup heap1, heap2 and initialize d1[][] and d2[][] for all the
              // grids on the two subtrees
              setupHeap(netID, edgeID, pq1, v2, regionX1, regionX2, regionY1,
                        regionY2, d1, corrEdge, inRegion);
              // TODO: use std priority queue
              // while loop to find shortest path
              ind1 = (pq1.top().d1_p - &d1[0][0]);
              pq1.pop();
              curX = ind1 % xGrid;
              curY = ind1 / xGrid;

              for (local_vec::iterator ii = v2.begin(); ii != v2.end(); ii++) {
                pop_heap2[*ii] = TRUE;
              }
              float curr_d1;
              while (pop_heap2[ind1] ==
                     FALSE) // stop until the grid position been popped out from
                            // both heap1 and heap2
              {
                // relax all the adjacent grids within the enlarged region for
                // source subtree

                // if(PRINT) printf("curX curY %d %d, (%d, %d), (%d, %d),
                // pq1.size: %d\n", curX, curY, regionX1, regionX2, regionY1,
                // regionY2, pq1.size()); if(curX == 102 && curY == 221)
                // exit(1);
                curr_d1 = d1[curY][curX];
                if (curr_d1 != 0) {
                  if (HV[curY][curX]) {
                    preX = parentX1[curY][curX];
                    preY = parentY1[curY][curX];
                  } else {
                    preX = parentX3[curY][curX];
                    preY = parentY3[curY][curX];
                  }
                } else {
                  preX = curX;
                  preY = curY;
                }

                // left
                if (curX > regionX1) {
                  grid = curY * (xGrid - 1) + curX - 1;
                  tmpX = curX - 1; // the left neighbor
                  if ((preY == curY) || (curr_d1 == 0)) {
                    tmp = curr_d1 +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                  } else {
                    if (curX < regionX2 - 1) {
                      tmp_grid = curY * (xGrid - 1) + curX;
                      tmp_cost =
                          d1[curY][curX + 1] +
                          h_costTable[h_edges[tmp_grid].usage +
                                      h_edges[tmp_grid].red +
                                      (int)(L * h_edges[tmp_grid].last_usage)];

                      if (tmp_cost < curr_d1 + VIA &&
                          d1[curY][tmpX] >
                              tmp_cost +
                                  h_costTable
                                      [h_edges[grid].usage + h_edges[grid].red +
                                       (int)(L * h_edges[grid].last_usage)]) {
                        hyperH[curY][curX] = TRUE; // Michael
                      }
                    }
                    tmp = curr_d1 + VIA +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                  }
                  // if(LOCK)  h_edges[grid].releaseLock();

                  if (d1[curY][tmpX] >
                      tmp) // left neighbor been put into heap1 but needs update
                  {
                    d1[curY][tmpX]       = tmp;
                    parentX3[curY][tmpX] = curX;
                    parentY3[curY][tmpX] = curY;
                    HV[curY][tmpX]       = FALSE;
                    pq1.push({&(d1[curY][tmpX]), tmp});
                  }
                }
                // right
                if (curX < regionX2) {
                  grid = curY * (xGrid - 1) + curX;

                  tmpX = curX + 1; // the right neighbor
                  if ((preY == curY) || (curr_d1 == 0)) {
                    tmp = curr_d1 +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                  } else {
                    if (curX > regionX1 + 1) {
                      tmp_grid = curY * (xGrid - 1) + curX - 1;
                      tmp_cost =
                          d1[curY][curX - 1] +
                          h_costTable[h_edges[tmp_grid].usage +
                                      h_edges[tmp_grid].red +
                                      (int)(L * h_edges[tmp_grid].last_usage)];

                      if (tmp_cost < curr_d1 + VIA &&
                          d1[curY][tmpX] >
                              tmp_cost +
                                  h_costTable
                                      [h_edges[grid].usage + h_edges[grid].red +
                                       (int)(L * h_edges[grid].last_usage)]) {
                        hyperH[curY][curX] = TRUE;
                      }
                    }
                    tmp = curr_d1 + VIA +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                  }

                  if (d1[curY][tmpX] > tmp) // right neighbor been put into
                                            // heap1 but needs update
                  {
                    d1[curY][tmpX]       = tmp;
                    parentX3[curY][tmpX] = curX;
                    parentY3[curY][tmpX] = curY;
                    HV[curY][tmpX]       = FALSE;
                    pq1.push({&(d1[curY][tmpX]), tmp});
                  }
                }
                // bottom
                if (curY > regionY1) {
                  grid = (curY - 1) * xGrid + curX;

                  tmpY = curY - 1; // the bottom neighbor
                  if ((preX == curX) || (curr_d1 == 0)) {
                    tmp = curr_d1 +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                  } else {
                    if (curY < regionY2 - 1) {
                      tmp_grid = curY * xGrid + curX;
                      tmp_cost =
                          d1[curY + 1][curX] +
                          v_costTable[v_edges[tmp_grid].usage +
                                      v_edges[tmp_grid].red +
                                      (int)(L * v_edges[tmp_grid].last_usage)];

                      if (tmp_cost < curr_d1 + VIA &&
                          d1[tmpY][curX] >
                              tmp_cost +
                                  v_costTable
                                      [v_edges[grid].usage + v_edges[grid].red +
                                       (int)(L * v_edges[grid].last_usage)]) {
                        hyperV[curY][curX] = TRUE;
                      }
                    }
                    tmp = curr_d1 + VIA +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                  }

                  if (d1[tmpY][curX] > tmp) // bottom neighbor been put into
                                            // heap1 but needs update
                  {
                    d1[tmpY][curX]       = tmp;
                    parentX1[tmpY][curX] = curX;
                    parentY1[tmpY][curX] = curY;
                    HV[tmpY][curX]       = TRUE;
                    pq1.push({&(d1[tmpY][curX]), tmp});
                  }
                }
                // top
                if (curY < regionY2) {
                  grid = curY * xGrid + curX;
                  tmpY = curY + 1; // the top neighbor

                  if ((preX == curX) || (curr_d1 == 0)) {
                    tmp = curr_d1 +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                  } else {
                    if (curY > regionY1 + 1) {
                      tmp_grid = (curY - 1) * xGrid + curX;
                      tmp_cost =
                          d1[curY - 1][curX] +
                          v_costTable[v_edges[tmp_grid].usage +
                                      v_edges[tmp_grid].red +
                                      (int)(L * v_edges[tmp_grid].last_usage)];

                      if (tmp_cost < curr_d1 + VIA &&
                          d1[tmpY][curX] >
                              tmp_cost +
                                  v_costTable
                                      [v_edges[grid].usage + v_edges[grid].red +
                                       (int)(L * v_edges[grid].last_usage)]) {
                        hyperV[curY][curX] = TRUE;
                      }
                    }
                    tmp = curr_d1 + VIA +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                  }
                  if (d1[tmpY][curX] >
                      tmp) // top neighbor been put into heap1 but needs update
                  {
                    d1[tmpY][curX]       = tmp;
                    parentX1[tmpY][curX] = curX;
                    parentY1[tmpY][curX] = curY;
                    HV[tmpY][curX]       = TRUE;
                    pq1.push({&(d1[tmpY][curX]), tmp});
                  }
                }

                // update ind1 and ind2 for next loop, Michael: need to check if
                // it is up-to-date value.
                float d1_push;
                do {
                  ind1    = pq1.top().d1_p - &d1[0][0];
                  d1_push = pq1.top().d1_push;
                  pq1.pop();
                  curX = ind1 % xGrid;
                  curY = ind1 / xGrid;
                } while (d1_push != d1[curY][curX]);
              } // while loop

              for (local_vec::iterator ii = v2.begin(); ii != v2.end(); ii++)
                pop_heap2[*ii] = FALSE;

              crossX = ind1 % xGrid;
              crossY = ind1 / xGrid;

              cnt  = 0;
              curX = crossX;
              curY = crossY;
              while (d1[curY][curX] != 0) // loop until reach subtree1
              {

                /*if(cnt > 2000) {
                    cerr << "Y: " << curY <<" X:" << curX << " hyperH: " <<
                hyperH[curY][curX]; cerr << " hyperV:" << hyperV[curY][curX] <<
                " HV: " << HV[curY][curX]; cerr << " d1: " << d1[curY][curX] <<
                endl; cerr << " deadloop return!" << endl; reInitTree(netID);
                    return;
                }*/

                hypered = FALSE;
                if (cnt != 0) {
                  if (curX != tmpX && hyperH[curY][curX]) {
                    curX    = 2 * curX - tmpX;
                    hypered = TRUE;
                  }
                  // printf("hyperV[153][134]: %d\n", hyperV[curY][curX]);
                  if (curY != tmpY && hyperV[curY][curX]) {
                    curY    = 2 * curY - tmpY;
                    hypered = TRUE;
                  }
                }
                tmpX = curX;
                tmpY = curY;
                if (!hypered) {
                  if (HV[tmpY][tmpX]) {
                    curY = parentY1[tmpY][tmpX];
                  } else {
                    curX = parentX3[tmpY][tmpX];
                  }
                }

                tmp_gridsX[cnt] = curX;
                tmp_gridsY[cnt] = curY;
                cnt++;
              }
              // reverse the grids on the path

              for (i = 0; i < cnt; i++) {
                tmpind    = cnt - 1 - i;
                gridsX[i] = tmp_gridsX[tmpind];
                gridsY[i] = tmp_gridsY[tmpind];
              }
              // add the connection point (crossX, crossY)
              gridsX[cnt] = crossX;
              gridsY[cnt] = crossY;
              cnt++;

              curX     = crossX;
              curY     = crossY;
              cnt_n1n2 = cnt;

              // change the tree structure according to the new routing for the
              // tree edge find E1 and E2, and the endpoints of the edges they
              // are on
              E1x = gridsX[0];
              E1y = gridsY[0];
              E2x = gridsX[cnt_n1n2 - 1];
              E2y = gridsY[cnt_n1n2 - 1];

              edge_n1n2 = edgeID;
              // if(netID == 252163 && edgeID == 51)
              //    printf("E1x: %d, E1y: %d, E2x: %d, E2y %d length: %d\n",
              //    E1x, E1y, E2x, E2y, cnt_n1n2);

              // (1) consider subtree1
              if (n1 >= deg && (E1x != n1x || E1y != n1y))
              // n1 is not a pin and E1!=n1, then make change to subtree1,
              // otherwise, no change to subtree1
              {
                // find the endpoints of the edge E1 is on
                endpt1 = treeedges[corrEdge[E1y][E1x]].n1;
                endpt2 = treeedges[corrEdge[E1y][E1x]].n2;

                // find A1, A2 and edge_n1A1, edge_n1A2
                if (treenodes[n1].nbr[0] == n2) {
                  A1        = treenodes[n1].nbr[1];
                  A2        = treenodes[n1].nbr[2];
                  edge_n1A1 = treenodes[n1].edge[1];
                  edge_n1A2 = treenodes[n1].edge[2];
                } else if (treenodes[n1].nbr[1] == n2) {
                  A1        = treenodes[n1].nbr[0];
                  A2        = treenodes[n1].nbr[2];
                  edge_n1A1 = treenodes[n1].edge[0];
                  edge_n1A2 = treenodes[n1].edge[2];
                } else {
                  A1        = treenodes[n1].nbr[0];
                  A2        = treenodes[n1].nbr[1];
                  edge_n1A1 = treenodes[n1].edge[0];
                  edge_n1A2 = treenodes[n1].edge[1];
                }

                if (endpt1 == n1 ||
                    endpt2 == n1) // E1 is on (n1, A1) or (n1, A2)
                {
                  // if E1 is on (n1, A2), switch A1 and A2 so that E1 is always
                  // on (n1, A1)
                  if (endpt1 == A2 || endpt2 == A2) {
                    tmpi      = A1;
                    A1        = A2;
                    A2        = tmpi;
                    tmpi      = edge_n1A1;
                    edge_n1A1 = edge_n1A2;
                    edge_n1A2 = tmpi;
                  }

                  // update route for edge (n1, A1), (n1, A2)
                  updateRouteType1(treenodes, n1, A1, A2, E1x, E1y, treeedges,
                                   edge_n1A1, edge_n1A2);
                  // update position for n1
                  treenodes[n1].x = E1x;
                  treenodes[n1].y = E1y;
                }    // if E1 is on (n1, A1) or (n1, A2)
                else // E1 is not on (n1, A1) or (n1, A2), but on (C1, C2)
                {
                  C1        = endpt1;
                  C2        = endpt2;
                  edge_C1C2 = corrEdge[E1y][E1x];

                  // update route for edge (n1, C1), (n1, C2) and (A1, A2)
                  updateRouteType2(treenodes, n1, A1, A2, C1, C2, E1x, E1y,
                                   treeedges, edge_n1A1, edge_n1A2, edge_C1C2);
                  // update position for n1
                  treenodes[n1].x = E1x;
                  treenodes[n1].y = E1y;
                  // update 3 edges (n1, A1)->(C1, n1), (n1, A2)->(n1, C2), (C1,
                  // C2)->(A1, A2)
                  edge_n1C1               = edge_n1A1;
                  treeedges[edge_n1C1].n1 = C1;
                  treeedges[edge_n1C1].n2 = n1;
                  edge_n1C2               = edge_n1A2;
                  treeedges[edge_n1C2].n1 = n1;
                  treeedges[edge_n1C2].n2 = C2;
                  edge_A1A2               = edge_C1C2;
                  treeedges[edge_A1A2].n1 = A1;
                  treeedges[edge_A1A2].n2 = A2;
                  // update nbr and edge for 5 nodes n1, A1, A2, C1, C2
                  // n1's nbr (n2, A1, A2)->(n2, C1, C2)
                  treenodes[n1].nbr[0]  = n2;
                  treenodes[n1].edge[0] = edge_n1n2;
                  treenodes[n1].nbr[1]  = C1;
                  treenodes[n1].edge[1] = edge_n1C1;
                  treenodes[n1].nbr[2]  = C2;
                  treenodes[n1].edge[2] = edge_n1C2;
                  // A1's nbr n1->A2
                  for (i = 0; i < 3; i++) {
                    if (treenodes[A1].nbr[i] == n1) {
                      treenodes[A1].nbr[i]  = A2;
                      treenodes[A1].edge[i] = edge_A1A2;
                      break;
                    }
                  }
                  // A2's nbr n1->A1
                  for (i = 0; i < 3; i++) {
                    if (treenodes[A2].nbr[i] == n1) {
                      treenodes[A2].nbr[i]  = A1;
                      treenodes[A2].edge[i] = edge_A1A2;
                      break;
                    }
                  }
                  // C1's nbr C2->n1
                  for (i = 0; i < 3; i++) {
                    if (treenodes[C1].nbr[i] == C2) {
                      treenodes[C1].nbr[i]  = n1;
                      treenodes[C1].edge[i] = edge_n1C1;
                      break;
                    }
                  }
                  // C2's nbr C1->n1
                  for (i = 0; i < 3; i++) {
                    if (treenodes[C2].nbr[i] == C1) {
                      treenodes[C2].nbr[i]  = n1;
                      treenodes[C2].edge[i] = edge_n1C2;
                      break;
                    }
                  }

                } // else E1 is not on (n1, A1) or (n1, A2), but on (C1, C2)
              }   // n1 is not a pin and E1!=n1

              // (2) consider subtree2

              if (n2 >= deg && (E2x != n2x || E2y != n2y))
              // n2 is not a pin and E2!=n2, then make change to subtree2,
              // otherwise, no change to subtree2
              {
                // find the endpoints of the edge E1 is on
                endpt1 = treeedges[corrEdge[E2y][E2x]].n1;
                endpt2 = treeedges[corrEdge[E2y][E2x]].n2;

                // find B1, B2
                if (treenodes[n2].nbr[0] == n1) {
                  B1        = treenodes[n2].nbr[1];
                  B2        = treenodes[n2].nbr[2];
                  edge_n2B1 = treenodes[n2].edge[1];
                  edge_n2B2 = treenodes[n2].edge[2];
                } else if (treenodes[n2].nbr[1] == n1) {
                  B1        = treenodes[n2].nbr[0];
                  B2        = treenodes[n2].nbr[2];
                  edge_n2B1 = treenodes[n2].edge[0];
                  edge_n2B2 = treenodes[n2].edge[2];
                } else {
                  B1        = treenodes[n2].nbr[0];
                  B2        = treenodes[n2].nbr[1];
                  edge_n2B1 = treenodes[n2].edge[0];
                  edge_n2B2 = treenodes[n2].edge[1];
                }

                if (endpt1 == n2 ||
                    endpt2 == n2) // E2 is on (n2, B1) or (n2, B2)
                {
                  // if E2 is on (n2, B2), switch B1 and B2 so that E2 is always
                  // on (n2, B1)
                  if (endpt1 == B2 || endpt2 == B2) {
                    tmpi      = B1;
                    B1        = B2;
                    B2        = tmpi;
                    tmpi      = edge_n2B1;
                    edge_n2B1 = edge_n2B2;
                    edge_n2B2 = tmpi;
                  }

                  // update route for edge (n2, B1), (n2, B2)
                  updateRouteType1(treenodes, n2, B1, B2, E2x, E2y, treeedges,
                                   edge_n2B1, edge_n2B2);

                  // update position for n2
                  treenodes[n2].x = E2x;
                  treenodes[n2].y = E2y;
                }    // if E2 is on (n2, B1) or (n2, B2)
                else // E2 is not on (n2, B1) or (n2, B2), but on (D1, D2)
                {
                  D1        = endpt1;
                  D2        = endpt2;
                  edge_D1D2 = corrEdge[E2y][E2x];

                  // update route for edge (n2, D1), (n2, D2) and (B1, B2)
                  updateRouteType2(treenodes, n2, B1, B2, D1, D2, E2x, E2y,
                                   treeedges, edge_n2B1, edge_n2B2, edge_D1D2);
                  // update position for n2
                  treenodes[n2].x = E2x;
                  treenodes[n2].y = E2y;
                  // update 3 edges (n2, B1)->(D1, n2), (n2, B2)->(n2, D2), (D1,
                  // D2)->(B1, B2)
                  edge_n2D1               = edge_n2B1;
                  treeedges[edge_n2D1].n1 = D1;
                  treeedges[edge_n2D1].n2 = n2;
                  edge_n2D2               = edge_n2B2;
                  treeedges[edge_n2D2].n1 = n2;
                  treeedges[edge_n2D2].n2 = D2;
                  edge_B1B2               = edge_D1D2;
                  treeedges[edge_B1B2].n1 = B1;
                  treeedges[edge_B1B2].n2 = B2;
                  // update nbr and edge for 5 nodes n2, B1, B2, D1, D2
                  // n1's nbr (n1, B1, B2)->(n1, D1, D2)
                  treenodes[n2].nbr[0]  = n1;
                  treenodes[n2].edge[0] = edge_n1n2;
                  treenodes[n2].nbr[1]  = D1;
                  treenodes[n2].edge[1] = edge_n2D1;
                  treenodes[n2].nbr[2]  = D2;
                  treenodes[n2].edge[2] = edge_n2D2;
                  // B1's nbr n2->B2
                  for (i = 0; i < 3; i++) {
                    if (treenodes[B1].nbr[i] == n2) {
                      treenodes[B1].nbr[i]  = B2;
                      treenodes[B1].edge[i] = edge_B1B2;
                      break;
                    }
                  }
                  // B2's nbr n2->B1
                  for (i = 0; i < 3; i++) {
                    if (treenodes[B2].nbr[i] == n2) {
                      treenodes[B2].nbr[i]  = B1;
                      treenodes[B2].edge[i] = edge_B1B2;
                      break;
                    }
                  }
                  // D1's nbr D2->n2
                  for (i = 0; i < 3; i++) {
                    if (treenodes[D1].nbr[i] == D2) {
                      treenodes[D1].nbr[i]  = n2;
                      treenodes[D1].edge[i] = edge_n2D1;
                      break;
                    }
                  }
                  // D2's nbr D1->n2
                  for (i = 0; i < 3; i++) {
                    if (treenodes[D2].nbr[i] == D1) {
                      treenodes[D2].nbr[i]  = n2;
                      treenodes[D2].edge[i] = edge_n2D2;
                      break;
                    }
                  }
                } // else E2 is not on (n2, B1) or (n2, B2), but on (D1, D2)
              }   // n2 is not a pin and E2!=n2

              // update route for edge (n1, n2) and edge usage

              // printf("update route? %d %d\n", netID, num_edges);
              if (treeedges[edge_n1n2].route.type == MAZEROUTE) {
                free(treeedges[edge_n1n2].route.gridsX);
                free(treeedges[edge_n1n2].route.gridsY);
              }
              treeedges[edge_n1n2].route.gridsX =
                  (short*)calloc(cnt_n1n2, sizeof(short));
              treeedges[edge_n1n2].route.gridsY =
                  (short*)calloc(cnt_n1n2, sizeof(short));
              treeedges[edge_n1n2].route.type     = MAZEROUTE;
              treeedges[edge_n1n2].route.routelen = cnt_n1n2 - 1;
              treeedges[edge_n1n2].len = ADIFF(E1x, E2x) + ADIFF(E1y, E2y);
              treeedges[edge_n1n2].n_ripups += 1;
              total_ripups += 1;
              max_ripups.update(treeedges[edge_n1n2].n_ripups);

              for (i = 0; i < cnt_n1n2; i++) {
                // printf("cnt_n1n2: %d\n", cnt_n1n2);
                treeedges[edge_n1n2].route.gridsX[i] = gridsX[i];
                treeedges[edge_n1n2].route.gridsY[i] = gridsY[i];
              }

              // update edge usage

              /*for(i=0; i<pre_length; i++)
              {
                  if(pre_gridsX[i]==pre_gridsX[i+1]) // a vertical edge
                  {
                      if(i != pre_length - 1)
                          min_y = min(pre_gridsY[i], pre_gridsY[i+1]);
                      else
                          min_y = pre_gridsY[i];
                      //v_edges[min_y*xGrid+gridsX[i]].usage += 1;
                      //galois::atomicAdd(v_edges[min_y*xGrid+gridsX[i]].usage,
              (short unsigned)1);
                      //printf("x y %d %d i %d \n", pre_gridsX[i], min_y, i);
                      v_edges[min_y*xGrid+pre_gridsX[i]].usage.fetch_sub((short
              int)1);
                      //if(v_edges[min_y*xGrid+pre_gridsX[i]].usage < 0)
              printf("V negative! %d \n", i);
                  }
                  else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
                  {
                      if(i != pre_length - 1)
                          min_x = min(pre_gridsX[i], pre_gridsX[i+1]);
                      else
                          min_x = pre_gridsX[i];
                      //h_edges[gridsY[i]*(xGrid-1)+min_x].usage += 1;
                      //galois::atomicAdd(h_edges[gridsY[i]*(xGrid-1)+min_x].usage,
              (short unsigned)1);
                      //printf("x y %d %d i %d\n", min_x, pre_gridsY[i], i);
                      h_edges[pre_gridsY[i]*(xGrid-1)+min_x].usage.fetch_sub((short
              int)1);
                      //if(h_edges[pre_gridsY[i]*(xGrid-1)+min_x].usage < 0)
              printf("H negative! %d \n", i);
                  }
              }*/

              for (i = 0; i < cnt_n1n2 - 1; i++) {
                if (gridsX[i] == gridsX[i + 1]) // a vertical edge
                {
                  min_y = min(gridsY[i], gridsY[i + 1]);
                  // v_edges[min_y*xGrid+gridsX[i]].usage += 1;
                  // galois::atomicAdd(v_edges[min_y*xGrid+gridsX[i]].usage,
                  // (short unsigned)1);
                  v_edges[min_y * xGrid + gridsX[i]].usage.fetch_add(
                      (short int)1);

                } else /// if(gridsY[i]==gridsY[i+1])// a horizontal edge
                {
                  min_x = min(gridsX[i], gridsX[i + 1]);
                  // h_edges[gridsY[i]*(xGrid-1)+min_x].usage += 1;
                  // galois::atomicAdd(h_edges[gridsY[i]*(xGrid-1)+min_x].usage,
                  // (short unsigned)1);
                  h_edges[gridsY[i] * (xGrid - 1) + min_x].usage.fetch_add(
                      (short int)1);
                }
              }
              /*if(LOCK){
                  for(i=0; i<cnt_n1n2-1; i++)
                  {
                      if(gridsX[i]==gridsX[i+1]) // a vertical edge
                      {
                          min_y = min(gridsY[i], gridsY[i+1]);
                          v_edges[min_y*xGrid+gridsX[i]].releaseLock();
                      }
                      else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
                      {
                          min_x = min(gridsX[i], gridsX[i+1]);
                          h_edges[gridsY[i]*(xGrid-1)+min_x].releaseLock();
                      }
                  }
              }*/
              if (checkRoute2DTree(netID)) {
                reInitTree(netID);
                return;
              }
            } // congested route
          }   // maze routing
        }     // loop edgeID
      },
      // galois::wl<galois::worklists::ParaMeter<>>(),
      galois::steal(),
      // galois::chunk_size<64>(),
      galois::loopname("net-level parallelism")); // galois::do_all

  printf("total ripups: %d max ripups: %d\n", total_ripups.reduce(),
         max_ripups.reduce());
  //}, "mazeroute vtune function");
  free(h_costTable);
  free(v_costTable);
}

void mazeRouteMSMD_block(int iter, int expand, float costHeight,
                         int ripup_threshold, int mazeedge_Threshold,
                         Bool Ordering, int cost_type,
                         galois::InsertBag<int>* net_shuffle) {
  // LOCK = 0;
  float forange;

  // allocate memory for distance and parent and pop_heap
  h_costTable = (float*)calloc(40 * hCapacity, sizeof(float));
  v_costTable = (float*)calloc(40 * vCapacity, sizeof(float));

  forange = 40 * hCapacity;

  if (cost_type == 2) {
    for (int i = 0; i < forange; i++) {
      if (i < hCapacity - 1)
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i - 1) * LOGIS_COF) + 1) + 1;
      else
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i - 1) * LOGIS_COF) + 1) + 1 +
            costHeight / slope * (i - hCapacity);
    }
    forange = 40 * vCapacity;
    for (int i = 0; i < forange; i++) {
      if (i < vCapacity - 1)
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i - 1) * LOGIS_COF) + 1) + 1;
      else
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i - 1) * LOGIS_COF) + 1) + 1 +
            costHeight / slope * (i - vCapacity);
    }
  } else {

    for (int i = 0; i < forange; i++) {
      if (i < hCapacity)
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) + 1;
      else
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) + 1 +
            costHeight / slope * (i - hCapacity);
    }
    forange = 40 * vCapacity;
    for (int i = 0; i < forange; i++) {
      if (i < vCapacity)
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i) * LOGIS_COF) + 1) + 1;
      else
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i) * LOGIS_COF) + 1) + 1 +
            costHeight / slope * (i - vCapacity);
    }
  }

  /*forange = yGrid*xGrid;
  for(i=0; i<forange; i++)
  {
      pop_heap2[i] = FALSE;
  } //Michael*/

  if (Ordering) {
    StNetOrder();
    // printf("order?\n");
  }

  galois::substrate::PerThreadStorage<THREAD_LOCAL_STORAGE>
      thread_local_storage{};
  // for(nidRPC=0; nidRPC<numValidNets; nidRPC++)//parallelize
  PerThread_PQ perthread_pq;
  PerThread_Vec perthread_vec;
  PRINT = 0;
  // galois::runtime::profileVtune( [&] (void) {
  /*std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(net_shuffle.begin(), net_shuffle.end(), g);

  galois::do_all(galois::iterate(net_shuffle), */
  // galois::do_all(galois::iterate(0, numValidNets), [&] (const auto nidRPC)

  galois::on_each(
      [&](const unsigned tid, const unsigned numT)
      // for(unsigned int nidRPC = 0; nidRPC < numValidNets; nidRPC++)
      {
        if (tid >= numT)
          return;
        for (const auto nidRPC : net_shuffle[tid]) {
          int grid, netID;

          // maze routing for multi-source, multi-destination
          Bool hypered, enter;
          int i, j, deg, edgeID, n1, n2, n1x, n1y, n2x, n2y, ymin, ymax, xmin,
              xmax, curX, curY, crossX, crossY, tmpX, tmpY, tmpi, min_x, min_y,
              num_edges;
          int regionX1, regionX2, regionY1, regionY2;
          int ind1, tmpind, gridsX[XRANGE], gridsY[YRANGE], tmp_gridsX[XRANGE],
              tmp_gridsY[YRANGE];
          int endpt1, endpt2, A1, A2, B1, B2, C1, C2, D1, D2, cnt, cnt_n1n2;
          int edge_n1n2, edge_n1A1, edge_n1A2, edge_n1C1, edge_n1C2, edge_A1A2,
              edge_C1C2;
          int edge_n2B1, edge_n2B2, edge_n2D1, edge_n2D2, edge_B1B2, edge_D1D2;
          int E1x, E1y, E2x, E2y;
          int tmp_grid, tmp_cost;
          int preX, preY, origENG, edgeREC;

          float tmp;
          TreeEdge *treeedges, *treeedge;
          TreeNode* treenodes;

          bool* pop_heap2 = thread_local_storage.getLocal()->pop_heap2;

          float** d1    = thread_local_storage.getLocal()->d1_p;
          bool** HV     = thread_local_storage.getLocal()->HV_p;
          bool** hyperV = thread_local_storage.getLocal()->hyperV_p;
          bool** hyperH = thread_local_storage.getLocal()->hyperH_p;

          short** parentX1 = thread_local_storage.getLocal()->parentX1_p;
          short** parentX3 = thread_local_storage.getLocal()->parentX3_p;
          short** parentY1 = thread_local_storage.getLocal()->parentY1_p;
          short** parentY3 = thread_local_storage.getLocal()->parentY3_p;

          int** corrEdge = thread_local_storage.getLocal()->corrEdge_p;

          OrderNetEdge* netEO = thread_local_storage.getLocal()->netEO_p;

          bool** inRegion = thread_local_storage.getLocal()->inRegion_p;

          local_pq pq1 = perthread_pq.get();
          local_vec v2 = perthread_vec.get();

          /*for(i=0; i<yGrid*xGrid; i++)
          {
              pop_heap2[i] = FALSE;
          } */

          /*for(int i=0; i<yGrid; i++)
          {
              for(int j=0; j<xGrid; j++)
                  inRegion[i][j] = FALSE;
          }*/
          // printf("hyperV[153][134]: %d %d %d\n", hyperV[153][134],
          // parentY1[153][134], parentX3[153][134]); printf("what is
          // happening?\n");

          if (Ordering) {
            netID = treeOrderCong[nidRPC].treeIndex;
          } else {
            netID = nidRPC;
          }

          deg = sttrees[netID].deg;

          origENG = expand;

          netedgeOrderDec(netID, netEO);

          treeedges = sttrees[netID].edges;
          treenodes = sttrees[netID].nodes;
          // loop for all the tree edges (2*deg-3)
          num_edges = 2 * deg - 3;

          for (edgeREC = 0; edgeREC < num_edges; edgeREC++) {
            edgeID   = netEO[edgeREC].edgeID;
            treeedge = &(treeedges[edgeID]);

            n1            = treeedge->n1;
            n2            = treeedge->n2;
            n1x           = treenodes[n1].x;
            n1y           = treenodes[n1].y;
            n2x           = treenodes[n2].x;
            n2y           = treenodes[n2].y;
            treeedge->len = ADIFF(n2x, n1x) + ADIFF(n2y, n1y);

            if (treeedge->len >
                mazeedge_Threshold) // only route the non-degraded edges (len>0)
            {

              enter = newRipupCheck(treeedge, ripup_threshold, netID, edgeID);

              // ripup the routing for the edge
              if (enter) {

                if (n1y <= n2y) {
                  ymin = n1y;
                  ymax = n2y;
                } else {
                  ymin = n2y;
                  ymax = n1y;
                }

                if (n1x <= n2x) {
                  xmin = n1x;
                  xmax = n2x;
                } else {
                  xmin = n2x;
                  xmax = n1x;
                }

                int enlarge =
                    min(origENG,
                        (iter / 6 + 3) *
                            treeedge->route
                                .routelen); // michael, this was global variable
                regionX1 = max(0, xmin - enlarge);
                regionX2 = min(xGrid - 1, xmax + enlarge);
                regionY1 = max(0, ymin - enlarge);
                regionY2 = min(yGrid - 1, ymax + enlarge);

                // initialize d1[][] and d2[][] as BIG_INT
                for (i = regionY1; i <= regionY2; i++) {
                  for (j = regionX1; j <= regionX2; j++) {
                    d1[i][j] = BIG_INT;
                    /*d2[i][j] = BIG_INT;
                    hyperH[i][j] = FALSE;
                    hyperV[i][j] = FALSE;*/
                  }
                }
                // memset(hyperH, 0, xGrid * yGrid * sizeof(bool));
                // memset(hyperV, 0, xGrid * yGrid * sizeof(bool));
                for (i = regionY1; i <= regionY2; i++) {
                  for (j = regionX1; j <= regionX2; j++) {
                    hyperH[i][j] = FALSE;
                  }
                }
                for (i = regionY1; i <= regionY2; i++) {
                  for (j = regionX1; j <= regionX2; j++) {
                    hyperV[i][j] = FALSE;
                  }
                }
                // TODO: use seperate loops

                // setup heap1, heap2 and initialize d1[][] and d2[][] for all
                // the grids on the two subtrees
                setupHeap(netID, edgeID, pq1, v2, regionX1, regionX2, regionY1,
                          regionY2, d1, corrEdge, inRegion);
                // TODO: use std priority queue
                // while loop to find shortest path
                ind1 = (pq1.top().d1_p - &d1[0][0]);
                pq1.pop();
                curX = ind1 % xGrid;
                curY = ind1 / xGrid;

                for (local_vec::iterator ii = v2.begin(); ii != v2.end();
                     ii++) {
                  pop_heap2[*ii] = TRUE;
                }
                while (pop_heap2[ind1] ==
                       FALSE) // stop until the grid position been popped out
                              // from both heap1 and heap2
                {
                  // relax all the adjacent grids within the enlarged region for
                  // source subtree

                  // if(PRINT) printf("curX curY %d %d, (%d, %d), (%d, %d),
                  // pq1.size: %d\n", curX, curY, regionX1, regionX2, regionY1,
                  // regionY2, pq1.size()); if(curX == 102 && curY == 221)
                  // exit(1);
                  float curr_d1 = d1[curY][curX];
                  if (curr_d1 != 0) {
                    if (HV[curY][curX]) {
                      preX = parentX1[curY][curX];
                      preY = parentY1[curY][curX];
                    } else {
                      preX = parentX3[curY][curX];
                      preY = parentY3[curY][curX];
                    }
                  } else {
                    preX = curX;
                    preY = curY;
                  }

                  // left
                  if (curX > regionX1) {
                    grid = curY * (xGrid - 1) + curX - 1;

                    if ((preY == curY) || (curr_d1 == 0)) {
                      tmp =
                          curr_d1 +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                    } else {
                      if (curX < regionX2 - 1) {
                        tmp_grid = curY * (xGrid - 1) + curX;
                        tmp_cost =
                            d1[curY][curX + 1] +
                            h_costTable[h_edges[tmp_grid].usage +
                                        h_edges[tmp_grid].red +
                                        (int)(L *
                                              h_edges[tmp_grid].last_usage)];

                        if (tmp_cost < curr_d1 + VIA) {
                          hyperH[curY][curX] = TRUE; // Michael
                        }
                      }
                      tmp =
                          curr_d1 + VIA +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                    }
                    // if(LOCK)  h_edges[grid].releaseLock();
                    tmpX = curX - 1; // the left neighbor

                    /*if(d1[curY][tmpX]>=BIG_INT) // left neighbor not been put
                    into heap1
                    {
                        d1[curY][tmpX] = tmp;
                        parentX3[curY][tmpX] = curX;
                        parentY3[curY][tmpX] = curY;
                        HV[curY][tmpX] = FALSE;
                        pq1.push(&(d1[curY][tmpX]));
                    }
                    else */
                    if (d1[curY][tmpX] > tmp) // left neighbor been put into
                                              // heap1 but needs update
                    {
                      d1[curY][tmpX]       = tmp;
                      parentX3[curY][tmpX] = curX;
                      parentY3[curY][tmpX] = curY;
                      HV[curY][tmpX]       = FALSE;
                      pq1.push({&(d1[curY][tmpX]), tmp});
                    }
                  }
                  // right
                  if (curX < regionX2) {
                    grid = curY * (xGrid - 1) + curX;

                    if ((preY == curY) || (curr_d1 == 0)) {
                      tmp =
                          curr_d1 +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                    } else {
                      if (curX > regionX1 + 1) {
                        tmp_grid = curY * (xGrid - 1) + curX - 1;
                        tmp_cost =
                            d1[curY][curX - 1] +
                            h_costTable[h_edges[tmp_grid].usage +
                                        h_edges[tmp_grid].red +
                                        (int)(L *
                                              h_edges[tmp_grid].last_usage)];

                        if (tmp_cost < curr_d1 + VIA) {
                          hyperH[curY][curX] = TRUE;
                        }
                      }
                      tmp =
                          curr_d1 + VIA +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                    }
                    // if(LOCK) h_edges[grid].releaseLock();
                    tmpX = curX + 1; // the right neighbor

                    /*if(d1[curY][tmpX]>=BIG_INT) // right neighbor not been put
                    into heap1
                    {
                        d1[curY][tmpX] = tmp;
                        parentX3[curY][tmpX] = curX;
                        parentY3[curY][tmpX] = curY;
                        HV[curY][tmpX] = FALSE;
                        pq1.push(&(d1[curY][tmpX]));

                    }
                    else */
                    if (d1[curY][tmpX] > tmp) // right neighbor been put into
                                              // heap1 but needs update
                    {
                      d1[curY][tmpX]       = tmp;
                      parentX3[curY][tmpX] = curX;
                      parentY3[curY][tmpX] = curY;
                      HV[curY][tmpX]       = FALSE;
                      pq1.push({&(d1[curY][tmpX]), tmp});
                    }
                  }
                  // bottom
                  if (curY > regionY1) {
                    grid = (curY - 1) * xGrid + curX;

                    if ((preX == curX) || (curr_d1 == 0)) {
                      tmp =
                          curr_d1 +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                    } else {
                      if (curY < regionY2 - 1) {
                        tmp_grid = curY * xGrid + curX;
                        tmp_cost =
                            d1[curY + 1][curX] +
                            v_costTable[v_edges[tmp_grid].usage +
                                        v_edges[tmp_grid].red +
                                        (int)(L *
                                              v_edges[tmp_grid].last_usage)];

                        if (tmp_cost < curr_d1 + VIA) {
                          hyperV[curY][curX] = TRUE;
                        }
                      }
                      tmp =
                          curr_d1 + VIA +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                    }
                    // if(LOCK) v_edges[grid].releaseLock();
                    tmpY = curY - 1; // the bottom neighbor

                    /*if(d1[tmpY][curX]>=BIG_INT) // bottom neighbor not been
                    put into heap1
                    {
                        d1[tmpY][curX] = tmp;
                        parentX1[tmpY][curX] = curX;
                        parentY1[tmpY][curX] = curY;
                        HV[tmpY][curX] = TRUE;
                        pq1.push(&(d1[tmpY][curX]));

                    }
                    else */
                    if (d1[tmpY][curX] > tmp) // bottom neighbor been put into
                                              // heap1 but needs update
                    {
                      d1[tmpY][curX]       = tmp;
                      parentX1[tmpY][curX] = curX;
                      parentY1[tmpY][curX] = curY;
                      HV[tmpY][curX]       = TRUE;
                      pq1.push({&(d1[tmpY][curX]), tmp});
                    }
                  }
                  // top
                  if (curY < regionY2) {
                    grid = curY * xGrid + curX;

                    if ((preX == curX) || (curr_d1 == 0)) {
                      tmp =
                          curr_d1 +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                    } else {
                      if (curY > regionY1 + 1) {
                        tmp_grid = (curY - 1) * xGrid + curX;
                        tmp_cost =
                            d1[curY - 1][curX] +
                            v_costTable[v_edges[tmp_grid].usage +
                                        v_edges[tmp_grid].red +
                                        (int)(L *
                                              v_edges[tmp_grid].last_usage)];

                        if (tmp_cost < curr_d1 + VIA) {
                          hyperV[curY][curX] = TRUE;
                        }
                      }
                      tmp =
                          curr_d1 + VIA +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                    }
                    // if(LOCK) v_edges[grid].releaseLock();
                    tmpY = curY + 1; // the top neighbor

                    /*if(d1[tmpY][curX]>=BIG_INT) // top neighbor not been put
                    into heap1
                    {
                        d1[tmpY][curX] = tmp;
                        parentX1[tmpY][curX] = curX;
                        parentY1[tmpY][curX] = curY;
                        HV[tmpY][curX] = TRUE;
                        pq1.push(&(d1[tmpY][curX]));
                    }
                    else*/
                    if (d1[tmpY][curX] > tmp) // top neighbor been put into
                                              // heap1 but needs update
                    {
                      d1[tmpY][curX]       = tmp;
                      parentX1[tmpY][curX] = curX;
                      parentY1[tmpY][curX] = curY;
                      HV[tmpY][curX]       = TRUE;
                      pq1.push({&(d1[tmpY][curX]), tmp});
                    }
                  }

                  // update ind1 and ind2 for next loop, Michael: need to check
                  // if it is up-to-date value.
                  float d1_push;
                  do {
                    ind1    = pq1.top().d1_p - &d1[0][0];
                    d1_push = pq1.top().d1_push;
                    pq1.pop();
                    curX = ind1 % xGrid;
                    curY = ind1 / xGrid;
                  } while (d1_push != d1[curY][curX]);
                } // while loop

                for (local_vec::iterator ii = v2.begin(); ii != v2.end(); ii++)
                  pop_heap2[*ii] = FALSE;

                crossX = ind1 % xGrid;
                crossY = ind1 / xGrid;

                cnt  = 0;
                curX = crossX;
                curY = crossY;
                while (d1[curY][curX] != 0) // loop until reach subtree1
                {
                  hypered = FALSE;
                  if (cnt != 0) {
                    if (curX != tmpX && hyperH[curY][curX]) {
                      curX    = 2 * curX - tmpX;
                      hypered = TRUE;
                    }
                    // printf("hyperV[153][134]: %d\n", hyperV[curY][curX]);
                    if (curY != tmpY && hyperV[curY][curX]) {
                      curY    = 2 * curY - tmpY;
                      hypered = TRUE;
                    }
                  }
                  tmpX = curX;
                  tmpY = curY;
                  if (!hypered) {
                    if (HV[tmpY][tmpX]) {
                      curY = parentY1[tmpY][tmpX];
                    } else {
                      curX = parentX3[tmpY][tmpX];
                    }
                  }

                  tmp_gridsX[cnt] = curX;
                  tmp_gridsY[cnt] = curY;
                  cnt++;
                }
                // reverse the grids on the path

                for (i = 0; i < cnt; i++) {
                  tmpind    = cnt - 1 - i;
                  gridsX[i] = tmp_gridsX[tmpind];
                  gridsY[i] = tmp_gridsY[tmpind];
                }
                // add the connection point (crossX, crossY)
                gridsX[cnt] = crossX;
                gridsY[cnt] = crossY;
                cnt++;

                curX     = crossX;
                curY     = crossY;
                cnt_n1n2 = cnt;

                // change the tree structure according to the new routing for
                // the tree edge find E1 and E2, and the endpoints of the edges
                // they are on
                E1x = gridsX[0];
                E1y = gridsY[0];
                E2x = gridsX[cnt_n1n2 - 1];
                E2y = gridsY[cnt_n1n2 - 1];

                edge_n1n2 = edgeID;

                // (1) consider subtree1
                if (n1 >= deg && (E1x != n1x || E1y != n1y))
                // n1 is not a pin and E1!=n1, then make change to subtree1,
                // otherwise, no change to subtree1
                {
                  // find the endpoints of the edge E1 is on
                  endpt1 = treeedges[corrEdge[E1y][E1x]].n1;
                  endpt2 = treeedges[corrEdge[E1y][E1x]].n2;

                  // find A1, A2 and edge_n1A1, edge_n1A2
                  if (treenodes[n1].nbr[0] == n2) {
                    A1        = treenodes[n1].nbr[1];
                    A2        = treenodes[n1].nbr[2];
                    edge_n1A1 = treenodes[n1].edge[1];
                    edge_n1A2 = treenodes[n1].edge[2];
                  } else if (treenodes[n1].nbr[1] == n2) {
                    A1        = treenodes[n1].nbr[0];
                    A2        = treenodes[n1].nbr[2];
                    edge_n1A1 = treenodes[n1].edge[0];
                    edge_n1A2 = treenodes[n1].edge[2];
                  } else {
                    A1        = treenodes[n1].nbr[0];
                    A2        = treenodes[n1].nbr[1];
                    edge_n1A1 = treenodes[n1].edge[0];
                    edge_n1A2 = treenodes[n1].edge[1];
                  }

                  if (endpt1 == n1 ||
                      endpt2 == n1) // E1 is on (n1, A1) or (n1, A2)
                  {
                    // if E1 is on (n1, A2), switch A1 and A2 so that E1 is
                    // always on (n1, A1)
                    if (endpt1 == A2 || endpt2 == A2) {
                      tmpi      = A1;
                      A1        = A2;
                      A2        = tmpi;
                      tmpi      = edge_n1A1;
                      edge_n1A1 = edge_n1A2;
                      edge_n1A2 = tmpi;
                    }

                    // update route for edge (n1, A1), (n1, A2)
                    updateRouteType1(treenodes, n1, A1, A2, E1x, E1y, treeedges,
                                     edge_n1A1, edge_n1A2);
                    // update position for n1
                    treenodes[n1].x = E1x;
                    treenodes[n1].y = E1y;
                  }    // if E1 is on (n1, A1) or (n1, A2)
                  else // E1 is not on (n1, A1) or (n1, A2), but on (C1, C2)
                  {
                    C1        = endpt1;
                    C2        = endpt2;
                    edge_C1C2 = corrEdge[E1y][E1x];

                    // update route for edge (n1, C1), (n1, C2) and (A1, A2)
                    updateRouteType2(treenodes, n1, A1, A2, C1, C2, E1x, E1y,
                                     treeedges, edge_n1A1, edge_n1A2,
                                     edge_C1C2);
                    // update position for n1
                    treenodes[n1].x = E1x;
                    treenodes[n1].y = E1y;
                    // update 3 edges (n1, A1)->(C1, n1), (n1, A2)->(n1, C2),
                    // (C1, C2)->(A1, A2)
                    edge_n1C1               = edge_n1A1;
                    treeedges[edge_n1C1].n1 = C1;
                    treeedges[edge_n1C1].n2 = n1;
                    edge_n1C2               = edge_n1A2;
                    treeedges[edge_n1C2].n1 = n1;
                    treeedges[edge_n1C2].n2 = C2;
                    edge_A1A2               = edge_C1C2;
                    treeedges[edge_A1A2].n1 = A1;
                    treeedges[edge_A1A2].n2 = A2;
                    // update nbr and edge for 5 nodes n1, A1, A2, C1, C2
                    // n1's nbr (n2, A1, A2)->(n2, C1, C2)
                    treenodes[n1].nbr[0]  = n2;
                    treenodes[n1].edge[0] = edge_n1n2;
                    treenodes[n1].nbr[1]  = C1;
                    treenodes[n1].edge[1] = edge_n1C1;
                    treenodes[n1].nbr[2]  = C2;
                    treenodes[n1].edge[2] = edge_n1C2;
                    // A1's nbr n1->A2
                    for (i = 0; i < 3; i++) {
                      if (treenodes[A1].nbr[i] == n1) {
                        treenodes[A1].nbr[i]  = A2;
                        treenodes[A1].edge[i] = edge_A1A2;
                        break;
                      }
                    }
                    // A2's nbr n1->A1
                    for (i = 0; i < 3; i++) {
                      if (treenodes[A2].nbr[i] == n1) {
                        treenodes[A2].nbr[i]  = A1;
                        treenodes[A2].edge[i] = edge_A1A2;
                        break;
                      }
                    }
                    // C1's nbr C2->n1
                    for (i = 0; i < 3; i++) {
                      if (treenodes[C1].nbr[i] == C2) {
                        treenodes[C1].nbr[i]  = n1;
                        treenodes[C1].edge[i] = edge_n1C1;
                        break;
                      }
                    }
                    // C2's nbr C1->n1
                    for (i = 0; i < 3; i++) {
                      if (treenodes[C2].nbr[i] == C1) {
                        treenodes[C2].nbr[i]  = n1;
                        treenodes[C2].edge[i] = edge_n1C2;
                        break;
                      }
                    }

                  } // else E1 is not on (n1, A1) or (n1, A2), but on (C1, C2)
                }   // n1 is not a pin and E1!=n1

                // (2) consider subtree2

                if (n2 >= deg && (E2x != n2x || E2y != n2y))
                // n2 is not a pin and E2!=n2, then make change to subtree2,
                // otherwise, no change to subtree2
                {
                  // find the endpoints of the edge E1 is on
                  endpt1 = treeedges[corrEdge[E2y][E2x]].n1;
                  endpt2 = treeedges[corrEdge[E2y][E2x]].n2;

                  // find B1, B2
                  if (treenodes[n2].nbr[0] == n1) {
                    B1        = treenodes[n2].nbr[1];
                    B2        = treenodes[n2].nbr[2];
                    edge_n2B1 = treenodes[n2].edge[1];
                    edge_n2B2 = treenodes[n2].edge[2];
                  } else if (treenodes[n2].nbr[1] == n1) {
                    B1        = treenodes[n2].nbr[0];
                    B2        = treenodes[n2].nbr[2];
                    edge_n2B1 = treenodes[n2].edge[0];
                    edge_n2B2 = treenodes[n2].edge[2];
                  } else {
                    B1        = treenodes[n2].nbr[0];
                    B2        = treenodes[n2].nbr[1];
                    edge_n2B1 = treenodes[n2].edge[0];
                    edge_n2B2 = treenodes[n2].edge[1];
                  }

                  if (endpt1 == n2 ||
                      endpt2 == n2) // E2 is on (n2, B1) or (n2, B2)
                  {
                    // if E2 is on (n2, B2), switch B1 and B2 so that E2 is
                    // always on (n2, B1)
                    if (endpt1 == B2 || endpt2 == B2) {
                      tmpi      = B1;
                      B1        = B2;
                      B2        = tmpi;
                      tmpi      = edge_n2B1;
                      edge_n2B1 = edge_n2B2;
                      edge_n2B2 = tmpi;
                    }

                    // update route for edge (n2, B1), (n2, B2)
                    updateRouteType1(treenodes, n2, B1, B2, E2x, E2y, treeedges,
                                     edge_n2B1, edge_n2B2);

                    // update position for n2
                    treenodes[n2].x = E2x;
                    treenodes[n2].y = E2y;
                  }    // if E2 is on (n2, B1) or (n2, B2)
                  else // E2 is not on (n2, B1) or (n2, B2), but on (D1, D2)
                  {
                    D1        = endpt1;
                    D2        = endpt2;
                    edge_D1D2 = corrEdge[E2y][E2x];

                    // update route for edge (n2, D1), (n2, D2) and (B1, B2)
                    updateRouteType2(treenodes, n2, B1, B2, D1, D2, E2x, E2y,
                                     treeedges, edge_n2B1, edge_n2B2,
                                     edge_D1D2);
                    // update position for n2
                    treenodes[n2].x = E2x;
                    treenodes[n2].y = E2y;
                    // update 3 edges (n2, B1)->(D1, n2), (n2, B2)->(n2, D2),
                    // (D1, D2)->(B1, B2)
                    edge_n2D1               = edge_n2B1;
                    treeedges[edge_n2D1].n1 = D1;
                    treeedges[edge_n2D1].n2 = n2;
                    edge_n2D2               = edge_n2B2;
                    treeedges[edge_n2D2].n1 = n2;
                    treeedges[edge_n2D2].n2 = D2;
                    edge_B1B2               = edge_D1D2;
                    treeedges[edge_B1B2].n1 = B1;
                    treeedges[edge_B1B2].n2 = B2;
                    // update nbr and edge for 5 nodes n2, B1, B2, D1, D2
                    // n1's nbr (n1, B1, B2)->(n1, D1, D2)
                    treenodes[n2].nbr[0]  = n1;
                    treenodes[n2].edge[0] = edge_n1n2;
                    treenodes[n2].nbr[1]  = D1;
                    treenodes[n2].edge[1] = edge_n2D1;
                    treenodes[n2].nbr[2]  = D2;
                    treenodes[n2].edge[2] = edge_n2D2;
                    // B1's nbr n2->B2
                    for (i = 0; i < 3; i++) {
                      if (treenodes[B1].nbr[i] == n2) {
                        treenodes[B1].nbr[i]  = B2;
                        treenodes[B1].edge[i] = edge_B1B2;
                        break;
                      }
                    }
                    // B2's nbr n2->B1
                    for (i = 0; i < 3; i++) {
                      if (treenodes[B2].nbr[i] == n2) {
                        treenodes[B2].nbr[i]  = B1;
                        treenodes[B2].edge[i] = edge_B1B2;
                        break;
                      }
                    }
                    // D1's nbr D2->n2
                    for (i = 0; i < 3; i++) {
                      if (treenodes[D1].nbr[i] == D2) {
                        treenodes[D1].nbr[i]  = n2;
                        treenodes[D1].edge[i] = edge_n2D1;
                        break;
                      }
                    }
                    // D2's nbr D1->n2
                    for (i = 0; i < 3; i++) {
                      if (treenodes[D2].nbr[i] == D1) {
                        treenodes[D2].nbr[i]  = n2;
                        treenodes[D2].edge[i] = edge_n2D2;
                        break;
                      }
                    }
                  } // else E2 is not on (n2, B1) or (n2, B2), but on (D1, D2)
                }   // n2 is not a pin and E2!=n2

                // update route for edge (n1, n2) and edge usage

                // printf("update route? %d %d\n", netID, num_edges);
                if (treeedges[edge_n1n2].route.type == MAZEROUTE) {
                  free(treeedges[edge_n1n2].route.gridsX);
                  free(treeedges[edge_n1n2].route.gridsY);
                }
                treeedges[edge_n1n2].route.gridsX =
                    (short*)calloc(cnt_n1n2, sizeof(short));
                treeedges[edge_n1n2].route.gridsY =
                    (short*)calloc(cnt_n1n2, sizeof(short));
                treeedges[edge_n1n2].route.type     = MAZEROUTE;
                treeedges[edge_n1n2].route.routelen = cnt_n1n2 - 1;
                treeedges[edge_n1n2].len = ADIFF(E1x, E2x) + ADIFF(E1y, E2y);

                for (i = 0; i < cnt_n1n2; i++) {
                  // printf("cnt_n1n2: %d\n", cnt_n1n2);
                  treeedges[edge_n1n2].route.gridsX[i] = gridsX[i];
                  treeedges[edge_n1n2].route.gridsY[i] = gridsY[i];
                }

                // update edge usage

                for (i = 0; i < cnt_n1n2 - 1; i++) {
                  if (gridsX[i] == gridsX[i + 1]) // a vertical edge
                  {
                    min_y = min(gridsY[i], gridsY[i + 1]);
                    // v_edges[min_y*xGrid+gridsX[i]].usage += 1;
                    // galois::atomicAdd(v_edges[min_y*xGrid+gridsX[i]].usage,
                    // (short unsigned)1);
                    v_edges[min_y * xGrid + gridsX[i]].usage.fetch_add(
                        (short unsigned)1, std::memory_order_relaxed);
                  } else /// if(gridsY[i]==gridsY[i+1])// a horizontal edge
                  {
                    min_x = min(gridsX[i], gridsX[i + 1]);
                    // h_edges[gridsY[i]*(xGrid-1)+min_x].usage += 1;
                    // galois::atomicAdd(h_edges[gridsY[i]*(xGrid-1)+min_x].usage,
                    // (short unsigned)1);
                    h_edges[gridsY[i] * (xGrid - 1) + min_x].usage.fetch_add(
                        (short unsigned)1, std::memory_order_relaxed);
                  }
                }
                /*if(LOCK){
                    for(i=0; i<cnt_n1n2-1; i++)
                    {
                        if(gridsX[i]==gridsX[i+1]) // a vertical edge
                        {
                            min_y = min(gridsY[i], gridsY[i+1]);
                            v_edges[min_y*xGrid+gridsX[i]].releaseLock();
                        }
                        else ///if(gridsY[i]==gridsY[i+1])// a horizontal edge
                        {
                            min_x = min(gridsX[i], gridsX[i+1]);
                            h_edges[gridsY[i]*(xGrid-1)+min_x].releaseLock();
                        }
                    }
                }*/
                if (checkRoute2DTree(netID)) {
                  reInitTree(netID);
                  return;
                }
              } // congested route

            } // maze routing
          }   // loop edgeID
        }
      },
      galois::steal(),
      // galois::chunk_size<32>(),
      galois::loopname("maze routing block")); // galois::do_all
  //}, "mazeroute vtune function");
  free(h_costTable);
  free(v_costTable);
}

int getOverflow2Dmaze(int* maxOverflow, int* tUsage) {
  int i, j, grid, overflow, max_overflow, H_overflow, max_H_overflow,
      V_overflow, max_V_overflow, numedges = 0;
  int total_usage, total_cap;

  // get overflow
  overflow = max_overflow = H_overflow = max_H_overflow = V_overflow =
      max_V_overflow                                    = 0;

  total_usage = 0;
  total_cap   = 0;

  //    fprintf(fph, "Horizontal Congestion\n");

  // int ripup_same = 0;
  // int ripup_diff = 0;

  for (i = 0; i < yGrid; i++) {
    for (j = 0; j < xGrid - 1; j++) {
      grid = i * (xGrid - 1) + j;
      total_usage += h_edges[grid].usage;
      overflow = h_edges[grid].usage - h_edges[grid].cap;
      total_cap += h_edges[grid].cap;
      if (overflow > 0) {
        H_overflow += overflow;
        max_H_overflow = max(max_H_overflow, overflow);
        numedges++;
        /*if(!h_edges[grid].ripups_cur_round)
        {
            h_edges[grid].max_have_rippedups = h_edges[grid].max_ripups.load();
        }*/
      }
      // h_edges[grid].ripups_cur_round = false;
    }
  }
  //    fprintf(fpv, "\nVertical Congestion\n");
  for (i = 0; i < yGrid - 1; i++) {
    for (j = 0; j < xGrid; j++) {
      grid = i * xGrid + j;
      total_usage += v_edges[grid].usage;
      overflow = v_edges[grid].usage - v_edges[grid].cap;
      total_cap += v_edges[grid].cap;
      if (overflow > 0) {
        V_overflow += overflow;
        max_V_overflow = max(max_V_overflow, overflow);
        numedges++;
        /*if(!v_edges[grid].ripups_cur_round)
        {
            v_edges[grid].max_have_rippedups = v_edges[grid].max_ripups.load();
        }*/
      }

      // v_edges[grid].ripups_cur_round = false;
    }
  }
  /*bitmap_image image(dimx,dimy);
  for(i=0; i<yGrid-1; i++)
  {
      for(j=0; j<xGrid - 1; j++)
      {
          int gridx = i*(xGrid-1)+j;
          int gridy = i*xGrid+j;

          int h_overflow = h_edges[gridx].usage - h_edges[gridx].cap;
          h_overflow = (h_overflow > 0)? h_overflow : 0;
          int v_overflow = v_edges[gridy].usage - v_edges[gridy].cap;
          v_overflow = (v_overflow > 0)? v_overflow : 0;

          overflow = h_overflow + v_overflow;
          if(overflow > 0)
          {
              float red = (overflow >= 1)? 0 : (255 - ((float)overflow/1) *
  255); int red_int = (int)red; image.set_pixel(j,i,255,(unsigned
  char)red_int,(unsigned char)red_int);
          }
          else
          {
              image.set_pixel(j,i,255,255,255);
          }

      }
  }
  std::string file_name = "route" + to_string(PRINT_HEAT) + ".bmp";
  image.save_image(file_name);
  PRINT_HEAT++;*/

  max_overflow  = max(max_H_overflow, max_V_overflow);
  totalOverflow = H_overflow + V_overflow;
  *maxOverflow  = max_overflow;

  printf("total Usage   : %d\n", (int)total_usage);
  printf("Max H Overflow: %d\n", max_H_overflow);
  printf("Max V Overflow: %d\n", max_V_overflow);
  printf("Max Overflow  : %d\n", max_overflow);
  printf("Num Overflow e: %d\n", numedges);
  printf("H   Overflow  : %d\n", H_overflow);
  printf("V   Overflow  : %d\n", V_overflow);
  printf("Final Overflow: %d\n\n", totalOverflow);

  *tUsage = total_usage;

  if (total_usage > 800000) {
    ahTH = 30;
  } else {
    ahTH = 20;
  }

  return (totalOverflow);
}

void checkUsageCorrectness() {
  int* vedge_usage = new int[xGrid * (yGrid - 1)];
  int* hedge_usage = new int[(xGrid - 1) * yGrid];
  memset(vedge_usage, 0, xGrid * (yGrid - 1) * sizeof(int));
  memset(hedge_usage, 0, (xGrid - 1) * yGrid * sizeof(int));

  for (int netID = 0; netID < numValidNets; netID++) {
    // maze routing for multi-source, multi-destination

    int deg, edgeID, n1, n2, n1x, n1y, n2x, n2y, num_edges;

    TreeEdge *treeedges, *treeedge;
    TreeNode* treenodes;

    deg = sttrees[netID].deg;

    treeedges = sttrees[netID].edges;
    treenodes = sttrees[netID].nodes;
    // loop for all the tree edges (2*deg-3)
    num_edges = 2 * deg - 3;

    for (edgeID = 0; edgeID < num_edges; edgeID++) {
      treeedge = &(treeedges[edgeID]);

      n1            = treeedge->n1;
      n2            = treeedge->n2;
      n1x           = treenodes[n1].x;
      n1y           = treenodes[n1].y;
      n2x           = treenodes[n2].x;
      n2y           = treenodes[n2].y;
      treeedge->len = ADIFF(n2x, n1x) + ADIFF(n2y, n1y);

      if (treeedge->len > 0) // only route the non-degraded edges (len>0)
      {
        if (treeedge->route.type == MAZEROUTE) {
          short* gridsX = treeedge->route.gridsX;
          short* gridsY = treeedge->route.gridsY;

          // std::cout << "net: " << netID << " route lenth: " <<
          // treeedge->route.routelen << std::endl;

          for (int i = 0; i < treeedge->route.routelen; i++) {
            // std::cout << "path: " << gridsX[i] << "," << gridsY[i] << "<->"
            // << gridsX[i + 1] << "," << gridsY[i + 1] << std::endl;
            if (gridsX[i] == gridsX[i + 1]) {
              int ymin = min(gridsY[i], gridsY[i + 1]);
              vedge_usage[ymin * xGrid + gridsX[i]]++;
            } else if (gridsY[i] == gridsY[i + 1]) {
              int xmin = min(gridsX[i], gridsX[i + 1]);
              hedge_usage[gridsY[i] * (xGrid - 1) + xmin]++;
            } else {
              std::cout << "usage correctness: net not connected!" << std::endl;
            }
          }
        } else {
          std::cout << "usage correctness: not maze route!" << std::endl;
          exit(1);
        }
      }
    }
  }
  int same = 0;
  int diff = 0;
  for (int i = 0; i < yGrid; i++) {
    for (int j = 0; j < xGrid - 1; j++) {
      int grid = i * (xGrid - 1) + j;
      if (hedge_usage[grid] == h_edges[grid].usage) {
        same++;
      } else {
        diff++;
        // std::cout << "h edge diff: " << j << ", " << i << " check: " <<
        // hedge_usage[grid] << " actual: " << h_edges[grid].usage << std::endl;
      }
    }
  }

  for (int i = 0; i < yGrid - 1; i++) {
    for (int j = 0; j < xGrid; j++) {
      int grid = i * xGrid + j;
      if (vedge_usage[grid] == v_edges[grid].usage) {
        same++;
      } else {
        diff++;
        // std::cout << "v edge diff: " << j << ", " << i << " check: " <<
        // vedge_usage[grid] << " actual: " << v_edges[grid].usage << std::endl;
      }
    }
  }

  std::cout << "same: " << same << " diff: " << diff << std::endl;

  delete[] vedge_usage;
  delete[] hedge_usage;
}

int getOverflow2D(int* maxOverflow) {
  int i, j, grid, overflow, max_overflow, H_overflow, max_H_overflow,
      V_overflow, max_V_overflow, numedges;
  int total_usage, total_cap, hCap, vCap;

  // get overflow
  overflow = max_overflow = H_overflow = max_H_overflow = V_overflow =
      max_V_overflow                                    = 0;
  hCap = vCap = numedges = 0;

  total_usage = 0;
  total_cap   = 0;
  //    fprintf(fph, "Horizontal Congestion\n");
  for (i = 0; i < yGrid; i++) {
    for (j = 0; j < xGrid - 1; j++) {
      grid = i * (xGrid - 1) + j;
      total_usage += h_edges[grid].est_usage;
      overflow = h_edges[grid].est_usage - h_edges[grid].cap;
      total_cap += h_edges[grid].cap;
      hCap += h_edges[grid].cap;
      if (overflow > 0) {
        H_overflow += overflow;
        max_H_overflow = max(max_H_overflow, overflow);
        numedges++;
      }
    }
  }
  //    fprintf(fpv, "\nVertical Congestion\n");
  for (i = 0; i < yGrid - 1; i++) {
    for (j = 0; j < xGrid; j++) {
      grid = i * xGrid + j;
      total_usage += v_edges[grid].est_usage;
      overflow = v_edges[grid].est_usage - v_edges[grid].cap;
      total_cap += v_edges[grid].cap;
      vCap += v_edges[grid].cap;
      if (overflow > 0) {
        V_overflow += overflow;
        max_V_overflow = max(max_V_overflow, overflow);
        numedges++;
      }
    }
  }

  max_overflow  = max(max_H_overflow, max_V_overflow);
  totalOverflow = H_overflow + V_overflow;
  *maxOverflow  = max_overflow;

  if (total_usage > 800000) {
    ahTH = 30;
  } else {
    ahTH = 20;
  }

  printf("total hCap    : %d\n", hCap);
  printf("total vCap    : %d\n", vCap);
  printf("total Usage   : %d\n", (int)total_usage);
  printf("Max H Overflow: %d\n", max_H_overflow);
  printf("Max V Overflow: %d\n", max_V_overflow);
  printf("Max Overflow  : %d\n", max_overflow);
  printf("Num Overflow e: %d\n", numedges);
  printf("H   Overflow  : %d\n", H_overflow);
  printf("V   Overflow  : %d\n", V_overflow);
  printf("Final Overflow: %d\n\n", totalOverflow);

  return (totalOverflow);
}

int getOverflow3D(void) {
  int i, j, k, grid, overflow, max_overflow, H_overflow, max_H_overflow,
      V_overflow, max_V_overflow;
  int cap;
  int total_usage;

  // get overflow
  overflow = max_overflow = H_overflow = max_H_overflow = V_overflow =
      max_V_overflow                                    = 0;

  total_usage = 0;
  cap         = 0;
  //    fprintf(fph, "Horizontal Congestion\n");

  for (k = 0; k < numLayers; k++) {
    for (i = 0; i < yGrid; i++) {
      for (j = 0; j < xGrid - 1; j++) {
        grid = i * (xGrid - 1) + j + k * (xGrid - 1) * yGrid;
        total_usage += h_edges3D[grid].usage;
        overflow = h_edges3D[grid].usage - h_edges3D[grid].cap;
        cap += h_edges3D[grid].cap;

        if (overflow > 0) {
          H_overflow += overflow;
          max_H_overflow = max(max_H_overflow, overflow);
        }
      }
    }
    for (i = 0; i < yGrid - 1; i++) {
      for (j = 0; j < xGrid; j++) {
        grid = i * xGrid + j + k * xGrid * (yGrid - 1);
        total_usage += v_edges3D[grid].usage;
        overflow = v_edges3D[grid].usage - v_edges3D[grid].cap;
        cap += v_edges3D[grid].cap;

        if (overflow > 0) {
          V_overflow += overflow;
          max_V_overflow = max(max_V_overflow, overflow);
        }
      }
    }
  }

  max_overflow  = max(max_H_overflow, max_V_overflow);
  totalOverflow = H_overflow + V_overflow;

  printf("total Usage   : %d\n", total_usage);
  printf("Total Capacity: %d\n", cap);
  printf("Max H Overflow: %d\n", max_H_overflow);
  printf("Max V Overflow: %d\n", max_V_overflow);
  printf("Max Overflow  : %d\n", max_overflow);
  printf("H   Overflow  : %d\n", H_overflow);
  printf("V   Overflow  : %d\n", V_overflow);
  printf("Final Overflow: %d\n\n", totalOverflow);

  return (total_usage);
}

int unsolved;

/*void initialCongestionHistory(int round)
{
    int i, j, grid;

    for(i=0; i<yGrid; i++)
    {
        for(j=0; j<xGrid-1; j++)
        {
            grid = i*(xGrid-1)+j;
            h_edges[grid].est_usage -=
((float)h_edges[grid].usage/h_edges[grid].cap);

        }
    }

    for(i=0; i<yGrid-1; i++)
    {
        for(j=0; j<xGrid; j++)
        {
            grid = i*xGrid+j;
            v_edges[grid].est_usage -=
((float)v_edges[grid].usage/v_edges[grid].cap);

        }
    }

}

void reduceCongestionHistory(int round)
{
    int i, j, grid;

    for(i=0; i<yGrid; i++)
    {
        for(j=0; j<xGrid-1; j++)
        {
            grid = i*(xGrid-1)+j;
            h_edges[grid].est_usage -=
0.2*((float)h_edges[grid].usage/h_edges[grid].cap);
        }
    }

    for(i=0; i<yGrid-1; i++)
    {
        for(j=0; j<xGrid; j++)
        {
            grid = i*xGrid+j;
            v_edges[grid].est_usage -=
0.2*((float)v_edges[grid].usage/v_edges[grid].cap);
        }
    }

}*/

void InitEstUsage() {
  int i, j, grid;
  for (i = 0; i < yGrid; i++) {
    for (j = 0; j < xGrid - 1; j++) {
      grid                    = i * (xGrid - 1) + j;
      h_edges[grid].est_usage = 0;
    }
  }
  //    fprintf(fpv, "\nVertical Congestion\n");
  for (i = 0; i < yGrid - 1; i++) {
    for (j = 0; j < xGrid; j++) {
      grid                    = i * xGrid + j;
      v_edges[grid].est_usage = 0;
    }
  }
}

void str_accu(int rnd) {
  int i, j, grid, overflow;
  for (i = 0; i < yGrid; i++) {
    for (j = 0; j < xGrid - 1; j++) {
      grid     = i * (xGrid - 1) + j;
      overflow = h_edges[grid].usage - h_edges[grid].cap;
      if (overflow > 0 || h_edges[grid].congCNT > rnd) {
        h_edges[grid].last_usage += h_edges[grid].congCNT * overflow / 2;
      }
    }
  }
  //    fprintf(fpv, "\nVertical Congestion\n");
  for (i = 0; i < yGrid - 1; i++) {
    for (j = 0; j < xGrid; j++) {
      grid     = i * xGrid + j;
      overflow = v_edges[grid].usage - v_edges[grid].cap;
      if (overflow > 0 || v_edges[grid].congCNT > rnd) {
        v_edges[grid].last_usage += v_edges[grid].congCNT * overflow / 2;
      }
    }
  }
}

void InitLastUsage(int upType) {
  int i, j, grid;
  for (i = 0; i < yGrid; i++) {
    for (j = 0; j < xGrid - 1; j++) {
      grid                     = i * (xGrid - 1) + j;
      h_edges[grid].last_usage = 0;
    }
  }
  //    fprintf(fpv, "\nVertical Congestion\n");
  for (i = 0; i < yGrid - 1; i++) {
    for (j = 0; j < xGrid; j++) {
      grid                     = i * xGrid + j;
      v_edges[grid].last_usage = 0;
    }
  }

  if (upType == 1) {
    for (i = 0; i < yGrid; i++) {
      for (j = 0; j < xGrid - 1; j++) {
        grid                  = i * (xGrid - 1) + j;
        h_edges[grid].congCNT = 0;
      }
    }
    //    fprintf(fpv, "\nVertical Congestion\n");
    for (i = 0; i < yGrid - 1; i++) {
      for (j = 0; j < xGrid; j++) {
        grid                  = i * xGrid + j;
        v_edges[grid].congCNT = 0;
      }
    }
  } else if (upType == 2) {

    for (i = 0; i < yGrid; i++) {
      for (j = 0; j < xGrid - 1; j++) {

        grid = i * (xGrid - 1) + j;
        // if (overflow > 0)
        h_edges[grid].last_usage = h_edges[grid].last_usage * 0.2;
      }
    }
    //    fprintf(fpv, "\nVertical Congestion\n");
    for (i = 0; i < yGrid - 1; i++) {
      for (j = 0; j < xGrid; j++) {

        grid = i * xGrid + j;
        //	if (overflow > 0)
        v_edges[grid].last_usage = v_edges[grid].last_usage * 0.2;
      }
    }
  }
}

#endif
