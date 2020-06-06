#ifndef _DATAPROC_H_
#define _DATAPROC_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DataType.h"

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/PriorityQueue.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/substrate/SimpleLock.h"
#include "galois/AtomicHelpers.h"
#include "galois/runtime/Profile.h"
#include "galois/LargeArray.h"

#include "DataType.h"
#include "flute.h"

using namespace std;

#define BUFFERSIZE 800
#define STRINGLEN 100
#define MAXNETDEG 1000
#define MAXEDGES 10000000

#define MAXLEN 20000

#define XRANGE 2536
#define YRANGE 2536

// Global variables
float round_avg_dist;
int round_avg_length;
int xGrid, yGrid, numGrids, numNets, vCapacity, hCapacity;
float vCapacity_lb, hCapacity_lb, vCapacity_ub, hCapacity_ub;
int MaxDegree;
int MinWidth[MAXLAYER], MinSpacing[MAXLAYER], ViaSpacing[MAXLAYER];
int xcorner, ycorner, wTile, hTile;
int enlarge, costheight, ripup_threshold, ahTH;
int numValidNets,
    numInvalidNets; // # nets need to be routed (having pins in different grids)
int numLayers;
int totalNumSeg;   // total # segments
int totalOverflow; // total # overflow
int mazeThreshold; // the wirelen threshold to do maze routing
Net** nets;
Net** invalid_nets;
Edge *h_edges, *v_edges;
float d1[YRANGE][XRANGE];
float d2[YRANGE][XRANGE];

/*Bool HV[YRANGE][XRANGE];
Bool hyperV[YRANGE][XRANGE];
Bool hyperH[YRANGE][XRANGE];

int corrEdge[YRANGE][XRANGE];*/ //Michael
int SLOPE;
int vCapacity3D[MAXLAYER], hCapacity3D[MAXLAYER];

float LB;
float UB;
int THRESH_M;
double LOGIS_COF;
int ENLARGE;
int STEP;
int COSHEIGHT;
int STOP;
int VCA;
float L;
int VIA, slope, max_adj;

char benchFile[STRINGLEN];

Segment* seglist;
int* seglistIndex; // the index for the segments for each net
int* seglistCnt;   // the number of segements for each net
int* segOrder;     // the order of segments for routing
Tree* trees;       // the tree topologies
StTree* sttrees;   // the Steiner trees
DTYPE** gxs;       // the copy of xs for nets, used for second FLUTE
DTYPE** gys;       // the copy of xs for nets, used for second FLUTE
DTYPE** gs; // the copy of vertical sequence for nets, used for second FLUTE
int MD = 0, TD = 0;

Edge3D* h_edges3D;
Edge3D* v_edges3D;

OrderNetPin* treeOrderPV;
OrderTree* treeOrderCong;
int numTreeedges;
int viacost;

int layerGrid[MAXLAYER][MAXLEN];
int gridD[MAXLAYER][MAXLEN];
int viaLink[MAXLAYER][MAXLEN];

int d13D[MAXLAYER][YRANGE][XRANGE];
short d23D[MAXLAYER][YRANGE][XRANGE];

dirctionT*** directions3D;
int*** corrEdge3D;
parent3D*** pr3D;

int mazeedge_Threshold;
Bool inRegion[YRANGE][XRANGE];

Bool heapVisited[MAXNETDEG];
// int heapQueue[MAXNETDEG]; //Michael

int gridHV, gridH, gridV, gridHs[MAXLAYER], gridVs[MAXLAYER];

int** heap13D;
short** heap23D;

float *h_costTable, *v_costTable;
Bool stopDEC, errorPRONE;
// OrderNetEdge netEO[2000]; //Michael
int xcor[2000], ycor[2000], dcor[2000];

StTree* sttreesBK;

short **parentX1, **parentY1, **parentX3, **parentY3;

/*float **heap2,**heap1; //Michael

Bool *pop_heap2;*/

// Michael:
int LOCK;

void readFile(const char* benchFile) {
  FILE* fp;
  int i, j, k;
  int pinX, pinY, pinL, netID, numPins, pinInd = 0, grid, newnetID,
                                        invalid_netID, segcount, minwidth;
  float pinX_in, pinY_in;
  int maxDeg = 0;
  int pinXarray[MAXNETDEG], pinYarray[MAXNETDEG], pinLarray[MAXNETDEG];
  char netName[STRINGLEN];
  Bool remove;
  int numAdjust, x1, x2, y1, y2, l1, l2, cap, reduce, reducedCap;
  int net;
  int TC;

  net = 0;

  fp = fopen(benchFile, "r");
  if (fp == NULL) {
    printf("Error in opening %s\n", benchFile);
    exit(1);
  }

  if (fscanf(fp, "grid	%d %d %d\n", &xGrid, &yGrid, &numLayers) != 3)
    abort_with_message("Failed to read required info from benchfile.");

  vCapacity = hCapacity = 0;

  if (fscanf(fp, "vertical capacity	"))
    abort_with_message("Failed to read required info from benchfile.");
  for (i = 0; i < numLayers; i++) {
    if (fscanf(fp, "%d", &vCapacity3D[i]) != 1)
      abort_with_message("Failed to read required info from benchfile.");
    if (fscanf(fp, " "))
      abort_with_message("Failed to read required info from benchfile.");
    vCapacity3D[i] = vCapacity3D[i] / 2;
    vCapacity += vCapacity3D[i];
  }
  if (fscanf(fp, "\n"))
    abort_with_message("Failed to read required info from benchfile.");

  if (fscanf(fp, "horizontal capacity	"))
    abort_with_message("Failed to read required info from benchfile.");
  for (i = 0; i < numLayers; i++) {
    if (fscanf(fp, "%d", &hCapacity3D[i]) != 1)
      abort_with_message("Failed to read required info from benchfile.");
    if (fscanf(fp, " "))
      abort_with_message("Failed to read required info from benchfile.");
    hCapacity3D[i] = hCapacity3D[i] / 2;
    hCapacity += hCapacity3D[i];
  }
  if (fscanf(fp, "\n"))
    abort_with_message("Failed to read required info from benchfile.");

  if (fscanf(fp, "minimum width	"))
    abort_with_message("Failed to read required info from benchfile.");
  for (i = 0; i < numLayers; i++) {
    if (fscanf(fp, "%d", &(MinWidth[i])) != 1)
      abort_with_message("Failed to read required info from benchfile.");
    if (fscanf(fp, " "))
      abort_with_message("Failed to read required info from benchfile.");
  }
  if (fscanf(fp, "\n"))
    abort_with_message("Failed to read required info from benchfile.");

  if (fscanf(fp, "minimum spacing	"))
    abort_with_message("Failed to read required info from benchfile.");
  for (i = 0; i < numLayers; i++) {
    if (fscanf(fp, "%d", &(MinSpacing[i])) != 1)
      abort_with_message("Failed to read required info from benchfile.");
    if (fscanf(fp, " "))
      abort_with_message("Failed to read required info from benchfile.");
  }
  if (fscanf(fp, "\n"))
    abort_with_message("Failed to read required info from benchfile.");

  if (fscanf(fp, "via spacing	"))
    abort_with_message("Failed to read required info from benchfile.");
  for (i = 0; i < numLayers; i++) {
    if (fscanf(fp, "%d", &(ViaSpacing[i])) != 1)
      abort_with_message("Failed to read required info from benchfile.");
    if (fscanf(fp, " "))
      abort_with_message("Failed to read required info from benchfile.");
  }
  if (fscanf(fp, "\n"))
    abort_with_message("Failed to read required info from benchfile.");

  if (fscanf(fp, "%d %d %d %d\n\n", &xcorner, &ycorner, &wTile, &hTile) != 4)
    abort_with_message("Failed to read required info from benchfile.");

  if (fscanf(fp, "num net %d\n", &numNets) != 1)
    abort_with_message("Failed to read required info from benchfile.");

  numGrids = xGrid * yGrid;

  vCapacity_lb = LB * vCapacity;
  hCapacity_lb = LB * hCapacity;
  vCapacity_ub = UB * vCapacity;
  hCapacity_ub = UB * hCapacity;

  printf("\n");
  printf("grid %d %d %d\n", xGrid, yGrid, numLayers);
  for (i = 0; i < numLayers; i++) {
    printf("Layer %d vertical capacity %d, horizontal capacity %d\n", i,
           vCapacity3D[i], hCapacity3D[i]);
  }

  printf("total vertical capacity %d\n", vCapacity);
  printf("total horizontal capacity %d\n", hCapacity);
  printf("num net %d\n", numNets);

  // allocate memory for nets
  nets         = (Net**)malloc(numNets * sizeof(Net*));
  invalid_nets = (Net**)malloc(numNets * sizeof(Net*));
  for (i = 0; i < numNets; i++) {
    nets[i]         = (Net*)malloc(sizeof(Net));
    invalid_nets[i] = (Net*)malloc(sizeof(Net));
  }
  seglistIndex = (int*)malloc((numNets + 1) * sizeof(int));

  // read nets information from the input file
  segcount      = 0;
  newnetID      = 0;
  invalid_netID = 0;
  for (i = 0; i < numNets; i++) {
    net++;
    if (fscanf(fp, "%s %d %d %d\n", netName, &netID, &numPins, &minwidth) != 4)
      abort_with_message("Failed to read required info from benchfile.");
    if (numPins < 1000) {
      pinInd = 0;
      for (j = 0; j < numPins; j++) {
        if (fscanf(fp, "%f	%f	%d\n", &pinX_in, &pinY_in, &pinL) != 3)
          abort_with_message("Failed to read required info from benchfile.");
        pinX = (int)((pinX_in - xcorner) / wTile);
        pinY = (int)((pinY_in - ycorner) / hTile);
        if (!(pinX < 0 || pinX >= xGrid || pinY < -1 || pinY >= yGrid ||
              pinL >= numLayers || pinL < 0)) {
          remove = FALSE;
          for (k = 0; k < pinInd; k++) {
            if (pinX == pinXarray[k] && pinY == pinYarray[k] &&
                pinL == pinLarray[k]) {
              remove = TRUE;
              break;
            }
          }
          if (!remove) // the pin is in different grid from other pins
          {
            pinXarray[pinInd] = pinX;
            pinYarray[pinInd] = pinY;
            pinLarray[pinInd] = pinL;
            pinInd++;
          }
        }
      }

      if (pinInd > 1) // valid net
      {
        MD = max(MD, pinInd);
        TD += pinInd;
        strcpy(nets[newnetID]->name, netName);
        nets[newnetID]->netIDorg = netID;
        nets[newnetID]->numPins  = numPins;
        nets[newnetID]->deg      = pinInd;
        nets[newnetID]->pinX     = (short*)malloc(pinInd * sizeof(short));
        nets[newnetID]->pinY     = (short*)malloc(pinInd * sizeof(short));
        nets[newnetID]->pinL     = (short*)malloc(pinInd * sizeof(short));

        for (j = 0; j < pinInd; j++) {
          nets[newnetID]->pinX[j] = pinXarray[j];
          nets[newnetID]->pinY[j] = pinYarray[j];
          nets[newnetID]->pinL[j] = pinLarray[j];
        }
        maxDeg                 = pinInd > maxDeg ? pinInd : maxDeg;
        seglistIndex[newnetID] = segcount;
        newnetID++;
        segcount +=
            2 * pinInd -
            3; // at most (2*numPins-2) nodes, (2*numPins-3) nets for a net
      }        // if valid net
      else {
        strcpy(invalid_nets[invalid_netID]->name, netName);
        invalid_nets[invalid_netID]->netIDorg = netID;
        invalid_nets[invalid_netID]->numPins  = numPins;
        invalid_nets[invalid_netID]->deg      = pinInd;
        invalid_nets[invalid_netID]->pinX =
            (short*)malloc(pinInd * sizeof(short));
        invalid_nets[invalid_netID]->pinY =
            (short*)malloc(pinInd * sizeof(short));
        invalid_nets[invalid_netID]->pinL =
            (short*)malloc(pinInd * sizeof(short));

        for (j = 0; j < pinInd; j++) {
          invalid_nets[invalid_netID]->pinX[j] = pinXarray[j];
          invalid_nets[invalid_netID]->pinY[j] = pinYarray[j];
          invalid_nets[invalid_netID]->pinL[j] = pinLarray[j];
        }
        invalid_netID++;
      }

    } // if

    else {
      for (j = 0; j < numPins; j++)
        if (fscanf(fp, "%f	%f	%d\n", &pinX_in, &pinY_in, &pinL) != 3)
          abort_with_message("Failed to read required info from benchfile.");
    }

  } // loop i
  printf("the total net number is %d\n\n", net);

  if ((pinInd > 1) && (pinInd < 1000)) {
    seglistIndex[newnetID] = segcount; // the end pointer of the seglist
  }
  numValidNets   = newnetID;
  numInvalidNets = invalid_netID;

  // allocate memory and initialize for edges

  h_edges = (Edge*)calloc(((xGrid - 1) * yGrid), sizeof(Edge));
  v_edges = (Edge*)calloc((xGrid * (yGrid - 1)), sizeof(Edge));

  v_edges3D = (Edge3D*)calloc((numLayers * xGrid * yGrid), sizeof(Edge3D));
  h_edges3D = (Edge3D*)calloc((numLayers * xGrid * yGrid), sizeof(Edge3D));

  // 2D edge innitialization
  TC = 0;
  for (i = 0; i < yGrid; i++) {
    for (j = 0; j < xGrid - 1; j++) {
      grid              = i * (xGrid - 1) + j;
      h_edges[grid].cap = hCapacity;
      TC += hCapacity;
      h_edges[grid].usage      = 0;
      h_edges[grid].est_usage  = 0;
      h_edges[grid].red        = 0;
      h_edges[grid].last_usage = 0;

      h_edges[grid].max_ripups         = 0;
      h_edges[grid].max_have_rippedups = 0;
      h_edges[grid].ripups_cur_round   = false;
    }
  }
  for (i = 0; i < yGrid - 1; i++) {
    for (j = 0; j < xGrid; j++) {
      grid              = i * xGrid + j;
      v_edges[grid].cap = vCapacity;
      TC += vCapacity;
      v_edges[grid].usage      = 0;
      v_edges[grid].est_usage  = 0;
      v_edges[grid].red        = 0;
      v_edges[grid].last_usage = 0;

      v_edges[grid].max_ripups         = 0;
      v_edges[grid].max_have_rippedups = 0;
      v_edges[grid].ripups_cur_round   = false;
    }
  }

  for (k = 0; k < numLayers; k++) {
    for (i = 0; i < yGrid; i++) {
      for (j = 0; j < xGrid - 1; j++) {
        grid                  = i * (xGrid - 1) + j + k * (xGrid - 1) * yGrid;
        h_edges3D[grid].cap   = hCapacity3D[k];
        h_edges3D[grid].usage = 0;
        h_edges3D[grid].red   = 0;
      }
    }
    for (i = 0; i < yGrid - 1; i++) {
      for (j = 0; j < xGrid; j++) {
        grid                  = i * xGrid + j + k * xGrid * (yGrid - 1);
        v_edges3D[grid].cap   = vCapacity3D[k];
        v_edges3D[grid].usage = 0;
        v_edges3D[grid].red   = 0;
      }
    }
  }

  // modify the capacity of edges according to the input file

  if (fscanf(fp, "%d\n", &numAdjust) != 1)
    abort_with_message("Failed to read required info from benchfile.");
  printf("num of Adjust is %d\n", numAdjust);
  while (numAdjust > 0) {
    if (fscanf(fp, "%d %d %d %d %d %d %d\n", &x1, &y1, &l1, &x2, &y2, &l2,
               &reducedCap) != 7)
      abort_with_message("Failed to read required info from benchfile.");
    reducedCap = reducedCap / 2;

    k = l1 - 1;

    if (y1 == y2) // horizontal edge
    {
      grid                = y1 * (xGrid - 1) + x1 + k * (xGrid - 1) * yGrid;
      cap                 = h_edges3D[grid].cap;
      reduce              = cap - reducedCap;
      h_edges3D[grid].cap = reducedCap;
      h_edges3D[grid].red = reduce;
      grid                = y1 * (xGrid - 1) + x1;
      h_edges[grid].cap -= reduce;
      h_edges[grid].red += reduce;

    } else if (x1 == x2) // vertical edge
    {
      grid                = y1 * xGrid + x1 + k * xGrid * (yGrid - 1);
      cap                 = v_edges3D[grid].cap;
      reduce              = cap - reducedCap;
      v_edges3D[grid].cap = reducedCap;
      v_edges3D[grid].red = reduce;
      grid                = y1 * xGrid + x1;
      v_edges[grid].cap -= reduce;
      v_edges[grid].red += reduce;
    }
    numAdjust--;
  }

  treeOrderCong = NULL;
  stopDEC       = FALSE;

  seglistCnt = (int*)malloc(numValidNets * sizeof(int));
  seglist    = (Segment*)malloc(segcount * sizeof(Segment));
  trees      = (Tree*)malloc(numValidNets * sizeof(Tree));
  sttrees    = (StTree*)malloc(numValidNets * sizeof(StTree));
  gxs        = (DTYPE**)malloc(numValidNets * sizeof(DTYPE*));
  gys        = (DTYPE**)malloc(numValidNets * sizeof(DTYPE*));
  gs         = (DTYPE**)malloc(numValidNets * sizeof(DTYPE*));

  gridHV = XRANGE * YRANGE;
  gridH  = (xGrid - 1) * yGrid;
  gridV  = xGrid * (yGrid - 1);
  for (k = 0; k < numLayers; k++) {
    gridHs[k] = k * gridH;
    gridVs[k] = k * gridV;
  }

  MaxDegree = MD;

  printf("# valid nets: %d\n", numValidNets);
  printf("# segments: %d\n", segcount);
  printf("maxDeg:     %d\n", maxDeg);
  printf("\nDone getting input\n");
  printf("MD: %d, AD: %.2f, #nets: %d, #routed nets: %d\n", MD,
         (float)TD / newnetID, numNets, newnetID);
  printf("TC is %d\n", TC);
  fclose(fp);

  /*parentX1 = (short**)calloc(yGrid, sizeof(short*));
  parentY1 = (short**)calloc(yGrid, sizeof(short*));
  parentX3 = (short**)calloc(yGrid, sizeof(short*));
  parentY3 = (short**)calloc(yGrid, sizeof(short*));


  for(i=0; i<yGrid; i++)
  {
      parentX1[i] = (short*)calloc(xGrid, sizeof(short));
      parentY1[i] = (short*)calloc(xGrid, sizeof(short));
      parentX3[i] = (short*)calloc(xGrid, sizeof(short));
      parentY3[i] = (short*)calloc(xGrid, sizeof(short));
  }*/

  /*pop_heap2 = (Bool*)calloc(yGrid*XRANGE, sizeof(Bool));

  // allocate memory for priority queue
  heap1 = (float**)calloc((yGrid*xGrid), sizeof(float*));
  heap2 = (float**)calloc((yGrid*xGrid), sizeof(float*));*/

  sttreesBK = NULL;
}

void init_usage() {
  int i;

  for (i = 0; i < yGrid * (xGrid - 1); i++)
    h_edges[i].usage = 0;
  for (i = 0; i < (yGrid - 1) * xGrid; i++)
    v_edges[i].usage = 0;
}

void freeAllMemory() {
  int deg, numEdges, edgeID;
  TreeEdge* treeedge;

  for (int i = 0; i < numValidNets; i++) {
    free(nets[i]->pinX);
    free(nets[i]->pinY);
    free(nets[i]->pinL);
    free(nets[i]);
  }
  free(seglistIndex);
  free(seglistCnt);
  free(seglist);
  free(h_edges);
  free(v_edges);
  free(h_edges3D);
  free(v_edges3D);
  free(segOrder);

  for (int i = 0; i < numValidNets; i++)
    free(trees[i].branch);
  free(trees);

  for (int i = 0; i < numValidNets; i++) {
    deg      = sttrees[i].deg;
    numEdges = 2 * deg - 3;
    for (edgeID = 0; edgeID < numEdges; edgeID++) {
      treeedge = &(sttrees[i].edges[edgeID]);
      if (treeedge->len > 0) {
        free(treeedge->route.gridsX);
        free(treeedge->route.gridsY);
        free(treeedge->route.gridsL);
      }
    }
    free(sttrees[i].nodes);
    free(sttrees[i].edges);
  }
  free(sttrees);

  /*for(i=0; i<yGrid; i++)
  {
      free(parentX1[i]);
      free(parentY1[i]);
      free(parentX3[i]);
      free(parentY3[i]);
  }
  free(parentX1);
  free(parentY1);
  free(parentX3);
  free(parentY3);
  free(pop_heap2);
  free(heap1);
  free(heap2); //Michael*/
}

#endif
