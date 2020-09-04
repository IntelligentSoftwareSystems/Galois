#include "galois/DynamicBitset.h"

struct grid_lock : public galois::runtime::Lockable {
public:
  int lock;
  int& getData() { return lock; }
};

int round_num = 0;

void mazeRouteMSMD_finegrain(int iter, int expand, float costHeight,
                             int ripup_threshold, int mazeedge_Threshold,
                             Bool Ordering, int cost_type) {
  // LOCK = 0;
  galois::StatTimer timer_finegrain("fine grain function", "fine grain maze");

  float forange;
  // std::cout << " enter here " << std::endl;
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
            (round_num > 50)
                ? 10
                : costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) +
                      1;
      else
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) + 1 +
            costHeight / slope * (i - hCapacity);
    }
    forange = 40 * vCapacity;
    for (int i = 0; i < forange; i++) {
      if (i < vCapacity)
        v_costTable[i] =
            (round_num > 50)
                ? 10
                : costHeight / (exp((float)(vCapacity - i) * LOGIS_COF) + 1) +
                      1;
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
  }*/

  // Michael

  galois::LargeArray<grid_lock> data;
  data.allocateInterleaved(yGrid * xGrid);
  for (int n = 0; n < yGrid * xGrid; n++) {
    data.constructAt(n);
  }

  if (Ordering) {
    StNetOrder();
    // printf("order?\n");
  }

  THREAD_LOCAL_STORAGE thread_local_storage{};
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
  galois::StatTimer timer_newripupcheck("ripup", "fine grain maze");
  galois::StatTimer timer_setupheap("setup heap", "fine grain maze");
  galois::StatTimer timer_traceback("trace back", "fine grain maze");
  galois::StatTimer timer_adjusttree("adjust tree", "fine grain maze");
  galois::StatTimer timer_updateusage("update usage", "fine grain maze");
  galois::StatTimer timer_checkroute2dtree("checkroute2dtree",
                                           "fine grain maze");
  galois::StatTimer timer_init("init", "fine grain maze");
  galois::StatTimer timer_foreach("foreach", "fine grain maze");
  for (int nidRPC = 0; nidRPC < numValidNets; nidRPC++) {

    int netID;

    // maze routing for multi-source, multi-destination
    Bool hypered, enter;
    int i, j, deg, edgeID, n1, n2, n1x, n1y, n2x, n2y, ymin, ymax, xmin, xmax,
        crossX, crossY, tmpi, min_x, min_y, num_edges;
    int regionX1, regionX2, regionY1, regionY2;
    int tmpind, gridsX[XRANGE], gridsY[YRANGE], tmp_gridsX[XRANGE],
        tmp_gridsY[YRANGE];
    int endpt1, endpt2, A1, A2, B1, B2, C1, C2, D1, D2, cnt, cnt_n1n2;
    int edge_n1n2, edge_n1A1, edge_n1A2, edge_n1C1, edge_n1C2, edge_A1A2,
        edge_C1C2;
    int edge_n2B1, edge_n2B2, edge_n2D1, edge_n2D2, edge_B1B2, edge_D1D2;
    int E1x, E1y, E2x, E2y;
    int origENG, edgeREC;

    TreeEdge *treeedges, *treeedge;
    TreeNode* treenodes;

    bool* pop_heap2 = thread_local_storage.pop_heap2;

    float** d1    = thread_local_storage.d1_p;
    bool** HV     = thread_local_storage.HV_p;
    bool** hyperV = thread_local_storage.hyperV_p;
    bool** hyperH = thread_local_storage.hyperH_p;

    short** parentX1 = thread_local_storage.parentX1_p;
    short** parentX3 = thread_local_storage.parentX3_p;
    short** parentY1 = thread_local_storage.parentY1_p;
    short** parentY3 = thread_local_storage.parentY3_p;

    int** corrEdge = thread_local_storage.corrEdge_p;

    OrderNetEdge* netEO = thread_local_storage.netEO_p;

    bool** inRegion = thread_local_storage.inRegion_p;

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
    // parentY1[153][134], parentX3[153][134]); printf("what is happening?\n");

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
        timer_newripupcheck.start();
        enter = newRipupCheck(treeedge, ripup_threshold, netID, edgeID);
        timer_newripupcheck.stop();

        // ripup the routing for the edge
        timer_finegrain.start();
        if (enter) {
          // if(netID == 2 && edgeID == 26)
          //    printf("netID %d edgeID %d src %d %d dst %d %d\n", netID,
          //    edgeID, n1x, n1y, n2x, n2y);
          // pre_length = treeedge->route.routelen;
          /*for(int i = 0; i < pre_length; i++)
          {
              pre_gridsY[i] = treeedge->route.gridsY[i];
              pre_gridsX[i] = treeedge->route.gridsX[i];
              //printf("i %d x %d y %d\n", i, pre_gridsX[i], pre_gridsY[i]);
          }*/
          timer_init.start();
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

          int enlarge = min(
              origENG, (iter / 6 + 3) *
                           treeedge->route
                               .routelen); // michael, this was global variable
          regionX1 = max(0, xmin - enlarge);
          regionX2 = min(xGrid - 1, xmax + enlarge);
          regionY1 = max(0, ymin - enlarge);
          regionY2 = min(yGrid - 1, ymax + enlarge);
          for (i = regionY1; i <= regionY2; i++) {
            for (j = regionX1; j <= regionX2; j++) {
              d1[i][j] = BIG_INT;
            }
          }
          for (i = regionY1; i <= regionY2; i++) {
            for (j = regionX1; j <= regionX2; j++) {
              HV[i][j] = FALSE;
            }
          }
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
          timer_setupheap.start();
          setupHeap(netID, edgeID, pq1, v2, regionX1, regionX2, regionY1,
                    regionY2, d1, corrEdge, inRegion);
          timer_setupheap.stop();
          // TODO: use std priority queue
          // while loop to find shortest path
          /*ind1 = (pq1.top().d1_p - &d1[0][0]);
          curX = ind1%xGrid;
          curY = ind1/xGrid;
          printf("src size: %d dst size: %d\n", pq1.size(), v2.size());*/
          for (local_vec::iterator ii = v2.begin(); ii != v2.end(); ii++) {
            pop_heap2[*ii] = TRUE;
          }
          std::atomic<int> return_ind1;
          std::atomic<float> return_dist;
          return_dist = (float)BIG_INT;

          timer_init.stop();
          timer_foreach.start();
          galois::for_each(
              galois::iterate(pq1),
              [&](const auto& top, auto& ctx)
              // while( pop_heap2[ind1]==FALSE) // stop until the grid position
              // been popped out from both heap1 and heap2
              {
                // relax all the adjacent grids within the enlarged region for
                // source subtree

                int ind1 = top.d1_p - &d1[0][0];

                float d1_push = top.d1_push;
                int curX      = ind1 % xGrid;
                int curY      = ind1 / xGrid;
                int grid      = curY * xGrid + curX;

                float curr_d1 = d1[curY][curX];
                // if(netID == 2 && edgeID == 26)
                //    printf("netID: %d edgeID:%d curX curY %d %d, d1_push: %f,
                //    curr_d1: %f\n", netID, edgeID, curX, curY, d1_push,
                //    curr_d1);

                if (d1_push > return_dist + OBIM_delta) {
                  // ctx.breakLoop();
                }
                galois::runtime::acquire(&data[grid],
                                         galois::MethodFlag::WRITE);
                if (d1_push == curr_d1 && d1_push < return_dist.load()) {
                  if (pop_heap2[ind1] != false) {
                    return_ind1.store(ind1);
                    return_dist.store(d1_push);
                  }

                  grid = curY * xGrid + curX - 1;
                  if (curX > regionX1)
                    galois::runtime::acquire(&data[grid],
                                             galois::MethodFlag::WRITE);

                  grid = curY * xGrid + curX + 1;
                  if (curX < regionX2)
                    galois::runtime::acquire(&data[grid],
                                             galois::MethodFlag::WRITE);

                  grid = (curY - 1) * xGrid + curX;
                  if (curY > regionY1)
                    galois::runtime::acquire(&data[grid],
                                             galois::MethodFlag::WRITE);

                  grid = (curY + 1) * xGrid + curX;
                  if (curY < regionY2)
                    galois::runtime::acquire(&data[grid],
                                             galois::MethodFlag::WRITE);

                  int preX, preY;
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
                  // printf("pop curY: %d curX: %d d1: %f preX: %d preY: %d
                  // hyperH: %d hyperV: %d HV: %d return_dist: %f\n",
                  //    curY, curX, curr_d1, preX, preY, hyperH[curY][curX],
                  //    hyperV[curY][curX], HV[curY][curX], return_dist.load());
                  float tmp, tmp_cost;
                  int tmp_grid;
                  int tmpX, tmpY;
                  // left
                  bool tmpH = false;
                  bool tmpV = false;
                  if (curX > regionX1) {
                    grid = curY * (xGrid - 1) + curX - 1;
                    // printf("grid: %d usage: %d red:%d last:%d sum%f
                    // %d\n",grid, h_edges[grid].usage.load(),
                    // h_edges[grid].red, h_edges[grid].last_usage, L ,
                    // h_edges[grid].usage.load() + h_edges[grid].red +
                    // (int)(L*h_edges[grid].last_usage));
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
                          // hyperH[curY][curX] = TRUE; //Michael
                          tmpH = true;
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
                    // galois::runtime::acquire(&data[curY * yGrid + tmpX],
                    // galois::MethodFlag::WRITE);
                    if (d1[curY][tmpX] > tmp &&
                        tmp < return_dist) // left neighbor been put into heap1
                                           // but needs update
                    {
                      d1[curY][tmpX]       = tmp;
                      parentX3[curY][tmpX] = curX;
                      parentY3[curY][tmpX] = curY;
                      HV[curY][tmpX]       = FALSE;
                      // pq1.push({&(d1[curY][tmpX]), tmp});
                      // pq_grid grid_push = {&(d1[curY][tmpX]), tmp};
                      ctx.push(pq_grid(&(d1[curY][tmpX]), tmp));
                      // printf("left push Y: %d X: %d tmp: %f HV: false hyperH:
                      // %d\n", curY, tmpX, tmp, true);
                    }
                  }
                  // right
                  if (curX < regionX2) {
                    grid = curY * (xGrid - 1) + curX;
                    // printf("grid: %d usage: %d red:%d last:%d sum%f
                    // %d\n",grid, h_edges[grid].usage.load(),
                    // h_edges[grid].red, h_edges[grid].last_usage, L ,
                    // h_edges[grid].usage.load() + h_edges[grid].red +
                    // (int)(L*h_edges[grid].last_usage));
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
                          // hyperH[curY][curX] = TRUE;
                          tmpH = true;
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
                    // galois::runtime::acquire(&data[curY * yGrid + tmpX],
                    // galois::MethodFlag::WRITE);
                    if (d1[curY][tmpX] > tmp &&
                        tmp < return_dist) // right neighbor been put into heap1
                                           // but needs update
                    {
                      d1[curY][tmpX]       = tmp;
                      parentX3[curY][tmpX] = curX;
                      parentY3[curY][tmpX] = curY;
                      HV[curY][tmpX]       = FALSE;
                      // pq1.push({&(d1[curY][tmpX]), tmp});
                      // pq_grid grid_push = {&(d1[curY][tmpX]), tmp};
                      ctx.push(pq_grid(&(d1[curY][tmpX]), tmp));
                      // printf("right push Y: %d X: %d tmp: %f HV: false
                      // hyperH: %d\n", curY, tmpX, tmp, true);
                    }
                  }
                  hyperH[curY][curX] = tmpH;

                  // bottom
                  if (curY > regionY1) {
                    grid = (curY - 1) * xGrid + curX;
                    // printf("grid: %d usage: %d red:%d last:%d sum%f
                    // %d\n",grid, v_edges[grid].usage.load(),
                    // v_edges[grid].red, v_edges[grid].last_usage, L ,
                    // v_edges[grid].usage.load() + v_edges[grid].red +
                    // (int)(L*v_edges[grid].last_usage));
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
                          // hyperV[curY][curX] = TRUE;
                          tmpV = true;
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
                    // galois::runtime::acquire(&data[tmpY * yGrid + curX],
                    // galois::MethodFlag::WRITE);
                    if (d1[tmpY][curX] > tmp &&
                        tmp < return_dist) // bottom neighbor been put into
                                           // heap1 but needs update
                    {
                      d1[tmpY][curX]       = tmp;
                      parentX1[tmpY][curX] = curX;
                      parentY1[tmpY][curX] = curY;
                      HV[tmpY][curX]       = TRUE;
                      // pq1.push({&(d1[tmpY][curX]), tmp});
                      // pq_grid grid_push = {&(d1[tmpY][curX]), tmp};
                      ctx.push(pq_grid(&(d1[tmpY][curX]), tmp));
                      // printf("bottom push Y: %d X: %d tmp: %f HV: false
                      // hyperH: %d\n", tmpY, curX, tmp, true);
                    }
                  }
                  // top
                  if (curY < regionY2) {
                    grid = curY * xGrid + curX;
                    // printf("grid: %d usage: %d red:%d last:%d sum%f
                    // %d\n",grid, v_edges[grid].usage.load(),
                    // v_edges[grid].red, v_edges[grid].last_usage, L ,
                    // v_edges[grid].usage.load() + v_edges[grid].red +
                    // (int)(L*v_edges[grid].last_usage));
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
                          // hyperV[curY][curX] = TRUE;
                          tmpV = true;
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
                    // galois::runtime::acquire(&data[tmpY * yGrid + curX],
                    // galois::MethodFlag::WRITE);
                    if (d1[tmpY][curX] > tmp &&
                        tmp < return_dist) // top neighbor been put into heap1
                                           // but needs update
                    {
                      d1[tmpY][curX]       = tmp;
                      parentX1[tmpY][curX] = curX;
                      parentY1[tmpY][curX] = curY;
                      HV[tmpY][curX]       = TRUE;
                      // pq_grid grid_push = {&(d1[tmpY][curX]), tmp};
                      ctx.push(pq_grid(&(d1[tmpY][curX]), tmp));
                      // printf("top push Y: %d X: %d tmp: %f HV: false hyperH:
                      // %d\n", tmpY, curX, tmp, true);
                      // pq1.push({&(d1[tmpY][curX]), tmp});
                    }
                  }
                  hyperV[curY][curX] = tmpV;
                }
              },
              galois::wl<galois::worklists::ParaMeter<>>(),
              // galois::wl<PSChunk>(),
              // galois::wl<OBIM>(RequestIndexer),
              // galois::chunk_size<MAZE_CHUNK_SIZE>()
              // galois::parallel_break(),
              // galois::steal(),
              galois::loopname("fine_grain"));

          timer_foreach.stop();

          for (local_vec::iterator ii = v2.begin(); ii != v2.end(); ii++)
            pop_heap2[*ii] = FALSE;

          crossX = return_ind1 % xGrid;
          crossY = return_ind1 / xGrid;

          cnt      = 0;
          int curX = crossX;
          int curY = crossY;
          int tmpX, tmpY;
          // if(netID == 2 && edgeID == 26)
          //    printf("crossX %d crossY %d return_d: %f\n", crossX, crossY,
          //    return_dist.load());
          timer_traceback.start();
          while (d1[curY][curX] != 0) // loop until reach subtree1
          {
            // if(cnt > 1000 && cnt < 1100)
            //    printf("Y: %d X: %d hyperH: %d hyperV: %d HV: %d d1: %f\n",
            //    curY, curX, hyperH[curY][curX], hyperV[curY][curX],
            //    HV[curY][curX], d1[curY][curX]);

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

          // change the tree structure according to the new routing for the tree
          // edge find E1 and E2, and the endpoints of the edges they are on
          E1x = gridsX[0];
          E1y = gridsY[0];
          E2x = gridsX[cnt_n1n2 - 1];
          E2y = gridsY[cnt_n1n2 - 1];

          edge_n1n2 = edgeID;

          timer_traceback.stop();

          // if(netID == 14628)
          //    printf("netID %d edgeID %d src %d %d dst %d %d routelen: %d\n",
          //    netID, edgeID, E1x, E1y, E2x, E2y, cnt_n1n2);
          // (1) consider subtree1
          timer_adjusttree.start();
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

            if (endpt1 == n1 || endpt2 == n1) // E1 is on (n1, A1) or (n1, A2)
            {
              // if E1 is on (n1, A2), switch A1 and A2 so that E1 is always on
              // (n1, A1)
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

            if (endpt1 == n2 || endpt2 == n2) // E2 is on (n2, B1) or (n2, B2)
            {
              // if E2 is on (n2, B2), switch B1 and B2 so that E2 is always on
              // (n2, B1)
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
          // std::cout << " adjsut tree" << std::endl;
          timer_adjusttree.stop();

          // update edge usage

          timer_updateusage.start();
          for (i = 0; i < cnt_n1n2 - 1; i++) {
            if (gridsX[i] == gridsX[i + 1]) // a vertical edge
            {
              min_y = min(gridsY[i], gridsY[i + 1]);
              // v_edges[min_y*xGrid+gridsX[i]].usage += 1;
              // galois::atomicAdd(v_edges[min_y*xGrid+gridsX[i]].usage, (short
              // unsigned)1);
              v_edges[min_y * xGrid + gridsX[i]].usage.fetch_add(
                  (short int)1, std::memory_order_relaxed);
            } else /// if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
              min_x = min(gridsX[i], gridsX[i + 1]);
              // h_edges[gridsY[i]*(xGrid-1)+min_x].usage += 1;
              // galois::atomicAdd(h_edges[gridsY[i]*(xGrid-1)+min_x].usage,
              // (short unsigned)1);
              h_edges[gridsY[i] * (xGrid - 1) + min_x].usage.fetch_add(
                  (short int)1, std::memory_order_relaxed);
            }
          }
          timer_updateusage.stop();
          timer_checkroute2dtree.start();
          if (checkRoute2DTree(netID)) {
            reInitTree(netID);
            return;
          }
          timer_checkroute2dtree.stop();
        } // congested route, if(enter)
        timer_finegrain.stop();
      } // only route the non-degraded edges (len>0)
    }   // iterate on edges of a net
  }

  printf("total ripups: %d max ripups: %d\n", total_ripups.reduce(),
         max_ripups.reduce());
  //}, "mazeroute vtune function");
  free(h_costTable);
  free(v_costTable);
}

void mazeRouteMSMD_finegrain_spinlock(int iter, int expand, float costHeight,
                                      int ripup_threshold,
                                      int mazeedge_Threshold, Bool Ordering,
                                      int cost_type) {
  // LOCK = 0;
  galois::StatTimer timer_finegrain("fine grain maze", "fine grain maze");

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
            (float)costHeight / slope * (i - hCapacity);
    }
    forange = 40 * vCapacity;
    for (int i = 0; i < forange; i++) {
      if (i < vCapacity - 1)
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i - 1) * LOGIS_COF) + 1) + 1;
      else
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i - 1) * LOGIS_COF) + 1) + 1 +
            (float)costHeight / slope * (i - vCapacity);
    }
  } else {

    for (int i = 0; i < forange; i++) {
      if (i < hCapacity)
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) + 1;
      else
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) + 1 +
            (float)costHeight / slope * (i - hCapacity);
    }
    forange = 40 * vCapacity;
    for (int i = 0; i < forange; i++) {
      if (i < vCapacity)
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i) * LOGIS_COF) + 1) + 1;
      else
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i) * LOGIS_COF) + 1) + 1 +
            (float)costHeight / slope * (i - vCapacity);
    }
  }

  // cout << " i = vCap:" << v_costTable[vCapacity - 1] << " "
  //     << v_costTable[vCapacity] << " " << v_costTable[vCapacity + 1] << endl;

  /*forange = yGrid*xGrid;
  for(i=0; i<forange; i++)
  {
      pop_heap2[i] = FALSE;
  } //Michael*/

  galois::LargeArray<galois::substrate::SimpleLock> data;
  data.allocateInterleaved(xGrid * yGrid);

  galois::substrate::SimpleLock return_lock;
  if (Ordering) {
    StNetOrder();
    // printf("order?\n");
  }

  THREAD_LOCAL_STORAGE thread_local_storage{};
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
  /*galois::StatTimer timer_newripupcheck("ripup", "fine grain maze");
  galois::StatTimer timer_setupheap("setup heap", "fine grain maze");
  galois::StatTimer timer_traceback("trace back", "fine grain maze");
  galois::StatTimer timer_adjusttree("adjust tree", "fine grain maze");
  galois::StatTimer timer_updateusage("update usage", "fine grain maze");
  galois::StatTimer timer_checkroute2dtree("checkroute2dtree", "fine grain
  maze"); galois::StatTimer timer_init("init", "fine grain maze");
  galois::StatTimer timer_foreach("foreach", "fine grain maze");
  galois::StatTimer timer_init_int("big int initialize", "fine grain maze");*/
  float acc_dist = 0;
  int acc_length = 0;
  int acc_cnt    = 0;
  float max_dist = 0;
  int max_length = 0;

  for (int nidRPC = 0; nidRPC < numValidNets; nidRPC++) {

    int netID;

    // maze routing for multi-source, multi-destination
    Bool hypered, enter;
    int i, j, deg, edgeID, n1, n2, n1x, n1y, n2x, n2y, ymin, ymax, xmin, xmax,
        crossX, crossY, tmpi, min_x, min_y, num_edges;
    int regionX1, regionX2, regionY1, regionY2;
    int tmpind, gridsX[XRANGE], gridsY[YRANGE], tmp_gridsX[XRANGE],
        tmp_gridsY[YRANGE];
    int endpt1, endpt2, A1, A2, B1, B2, C1, C2, D1, D2, cnt, cnt_n1n2;
    int edge_n1n2, edge_n1A1, edge_n1A2, edge_n1C1, edge_n1C2, edge_A1A2,
        edge_C1C2;
    int edge_n2B1, edge_n2B2, edge_n2D1, edge_n2D2, edge_B1B2, edge_D1D2;
    int E1x, E1y, E2x, E2y;
    int origENG, edgeREC;

    TreeEdge *treeedges, *treeedge;
    TreeNode* treenodes;

    bool* pop_heap2 = thread_local_storage.pop_heap2;

    float** d1    = thread_local_storage.d1_p;
    bool** HV     = thread_local_storage.HV_p;
    bool** hyperV = thread_local_storage.hyperV_p;
    bool** hyperH = thread_local_storage.hyperH_p;

    short** parentX1 = thread_local_storage.parentX1_p;
    short** parentX3 = thread_local_storage.parentX3_p;
    short** parentY1 = thread_local_storage.parentY1_p;
    short** parentY3 = thread_local_storage.parentY3_p;

    int** corrEdge = thread_local_storage.corrEdge_p;

    OrderNetEdge* netEO = thread_local_storage.netEO_p;

    bool** inRegion = thread_local_storage.inRegion_p;

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
    // parentY1[153][134], parentX3[153][134]); printf("what is happening?\n");

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
        // timer_newripupcheck.start();
        enter = newRipupCheck(treeedge, ripup_threshold, netID, edgeID);
        // timer_newripupcheck.stop();

        // ripup the routing for the edge
        timer_finegrain.start();
        if (enter) {
          // if(netID == 2 && edgeID == 26)
          //    printf("netID %d edgeID %d src %d %d dst %d %d\n", netID,
          //    edgeID, n1x, n1y, n2x, n2y);
          // pre_length = treeedge->route.routelen;
          /*for(int i = 0; i < pre_length; i++)
          {
              pre_gridsY[i] = treeedge->route.gridsY[i];
              pre_gridsX[i] = treeedge->route.gridsX[i];
              //printf("i %d x %d y %d\n", i, pre_gridsX[i], pre_gridsY[i]);
          }*/
          // timer_init.start();
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

          int enlarge = min(
              origENG, (iter / 6 + 3) *
                           treeedge->route
                               .routelen); // michael, this was global variable
          regionX1 = max(0, xmin - enlarge);
          regionX2 = min(xGrid - 1, xmax + enlarge);
          regionY1 = max(0, ymin - enlarge);
          regionY2 = min(yGrid - 1, ymax + enlarge);
          // std::cout << "region size" << regionWidth << ", " << regionHeight
          // << std::endl;
          // initialize d1[][] and d2[][] as BIG_INT
          // timer_init_int.start();
          for (i = regionY1; i <= regionY2; i++) {
            for (j = regionX1; j <= regionX2; j++) {
              d1[i][j] = BIG_INT;

              /*d2[i][j] = BIG_INT;
              hyperH[i][j] = FALSE;
              hyperV[i][j] = FALSE;*/
            }
          }
          // timer_init_int.stop();
          // memset(hyperH, 0, xGrid * yGrid * sizeof(bool));
          // memset(hyperV, 0, xGrid * yGrid * sizeof(bool));
          for (i = regionY1; i <= regionY2; i++) {
            for (j = regionX1; j <= regionX2; j++) {
              HV[i][j] = FALSE;
            }
          }
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
          // timer_setupheap.start();
          setupHeap(netID, edgeID, pq1, v2, regionX1, regionX2, regionY1,
                    regionY2, d1, corrEdge, inRegion);
          // timer_setupheap.stop();
          // TODO: use std priority queue
          // while loop to find shortest path
          /*ind1 = (pq1.top().d1_p - &d1[0][0]);
          curX = ind1%xGrid;
          curY = ind1/xGrid;
          printf("src size: %d dst size: %d\n", pq1.size(), v2.size());*/
          for (local_vec::iterator ii = v2.begin(); ii != v2.end(); ii++) {
            pop_heap2[*ii] = TRUE;
          }
          std::atomic<int> return_ind1;
          std::atomic<float> return_dist;
          return_dist = (float)BIG_INT;

          galois::for_each(
              galois::iterate(pq1),
              [&](const auto& top, auto& ctx)
              // while( pop_heap2[ind1]==FALSE) // stop until the grid position
              // been popped out from both heap1 and heap2
              {
                // relax all the adjacent grids within the enlarged region for
                // source subtree

                int ind1 = top.d1_p - &d1[0][0];

                int curX = ind1 % xGrid;
                int curY = ind1 / xGrid;
                int grid = curY * xGrid + curX;

                float curr_d1 = d1[curY][curX];
                float d1_push = top.d1_push;

                if (d1_push > return_dist + OBIM_delta) {
                  // ctx.breakLoop();
                }
                if (d1_push == curr_d1 && d1_push < return_dist.load()) {
                  if (pop_heap2[ind1] != false) {
                    // if(netID == 2 && edgeID == 26)
                    //    printf("reach! curX curY %d %d, d1_push: %f, curr_d1:
                    //    %f return_d: %f\n", curX, curY, d1_push, curr_d1,
                    //    return_dist.load());
                    return_lock.lock();
                    if (d1_push < return_dist.load()) {
                      return_ind1.store(ind1);
                      return_dist.store(d1_push);
                    } else {
                      return_lock.unlock();
                      return;
                    }
                    return_lock.unlock();
                  }
                  // curr_d1 = d1_push;

                  int preX = curX, preY = curY;
                  if (curr_d1 != 0) {
                    if (HV[curY][curX]) {
                      preX = parentX1[curY][curX];
                      preY = parentY1[curY][curX];
                    } else {
                      preX = parentX3[curY][curX];
                      preY = parentY3[curY][curX];
                    }
                  }
                  // printf("pop curY: %d curX: %d d1: %f preX: %d preY: %d
                  // hyperH: %d hyperV: %d HV: %d return_dist: %f\n",
                  //    curY, curX, curr_d1, preX, preY, hyperH[curY][curX],
                  //    hyperV[curY][curX], HV[curY][curX], return_dist.load());
                  float tmp = 0.f, tmp_cost = 0.f;
                  int tmp_grid = 0;
                  int tmpX = 0, tmpY = 0;
                  // left
                  bool tmpH = false;
                  bool tmpV = false;

                  if (curX > regionX1) {
                    grid = curY * (xGrid - 1) + curX - 1;

                    tmpX = curX - 1; // the left neighbor
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

                        if (tmp_cost < curr_d1 + VIA &&
                            d1[curY][tmpX] >
                                tmp_cost +
                                    h_costTable[h_edges[grid].usage +
                                                h_edges[grid].red +
                                                (int)(L * h_edges[grid]
                                                              .last_usage)]) {
                          tmpH = true;
                        }
                      }
                      tmp =
                          curr_d1 + VIA +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                    }

                    if (d1[curY][tmpX] > tmp && tmp < return_dist) {

                      data[curY * xGrid + curX - 1].lock();
                      if (d1[curY][tmpX] > tmp &&
                          tmp < return_dist) // left neighbor been put into
                                             // heap1 but needs update
                      {
                        d1[curY][tmpX]       = tmp;
                        parentX3[curY][tmpX] = curX;
                        parentY3[curY][tmpX] = curY;
                        HV[curY][tmpX]       = FALSE;
                        ctx.push(pq_grid(&(d1[curY][tmpX]), tmp));
                      }
                      data[curY * xGrid + curX - 1].unlock();
                    }
                  }
                  // right

                  if (curX < regionX2) {
                    grid = curY * (xGrid - 1) + curX;
                    tmpX = curX + 1; // the right neighbor
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

                        if (tmp_cost < curr_d1 + VIA &&
                            d1[curY][tmpX] >
                                tmp_cost +
                                    h_costTable[h_edges[grid].usage +
                                                h_edges[grid].red +
                                                (int)(L * h_edges[grid]
                                                              .last_usage)]) {
                          // hyperH[curY][curX] = TRUE;
                          tmpH = true;
                        }
                      }
                      tmp =
                          curr_d1 + VIA +
                          h_costTable[h_edges[grid].usage + h_edges[grid].red +
                                      (int)(L * h_edges[grid].last_usage)];
                    }

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
                    // galois::runtime::acquire(&data[curY * yGrid + tmpX],
                    // galois::MethodFlag::WRITE);
                    if (d1[curY][tmpX] > tmp && tmp < return_dist) {

                      data[curY * xGrid + curX + 1].lock();
                      if (d1[curY][tmpX] > tmp &&
                          tmp < return_dist) // right neighbor been put into
                                             // heap1 but needs update
                      {
                        d1[curY][tmpX]       = tmp;
                        parentX3[curY][tmpX] = curX;
                        parentY3[curY][tmpX] = curY;
                        HV[curY][tmpX]       = FALSE;
                        ctx.push(pq_grid(&(d1[curY][tmpX]), tmp));

                        // printf("right push Y: %d X: %d tmp: %f HV: false
                        // hyperH: %d\n", curY, tmpX, tmp, true);
                      }
                      data[curY * xGrid + curX + 1].unlock();
                    }
                  }
                  hyperH[curY][curX] = tmpH;

                  if (curY > regionY1) {
                    grid = (curY - 1) * xGrid + curX;

                    tmpY = curY - 1; // the bottom neighbor
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

                        if (tmp_cost < curr_d1 + VIA &&
                            d1[tmpY][curX] >
                                tmp_cost +
                                    v_costTable[v_edges[grid].usage +
                                                v_edges[grid].red +
                                                (int)(L * v_edges[grid]
                                                              .last_usage)]) {
                          // hyperV[curY][curX] = TRUE;
                          tmpV = true;
                        }
                      }
                      tmp =
                          curr_d1 + VIA +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                    }

                    if (d1[tmpY][curX] > tmp && tmp < return_dist) {

                      data[(curY - 1) * xGrid + curX].lock();
                      if (d1[tmpY][curX] > tmp &&
                          tmp < return_dist) // bottom neighbor been put into
                                             // heap1 but needs update
                      {
                        d1[tmpY][curX]       = tmp;
                        parentX1[tmpY][curX] = curX;
                        parentY1[tmpY][curX] = curY;
                        HV[tmpY][curX]       = TRUE;
                        ctx.push(pq_grid(&(d1[tmpY][curX]), tmp));
                      }
                      data[(curY - 1) * xGrid + curX].unlock();
                    }
                  }
                  // top
                  if (curY < regionY2) {

                    grid = curY * xGrid + curX;

                    tmpY = curY + 1; // the top neighbor
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

                        if (tmp_cost < curr_d1 + VIA &&
                            d1[tmpY][curX] >
                                tmp_cost +
                                    v_costTable[v_edges[grid].usage +
                                                v_edges[grid].red +
                                                (int)(L * v_edges[grid]
                                                              .last_usage)]) {
                          // hyperV[curY][curX] = TRUE;
                          tmpV = true;
                        }
                      }
                      tmp =
                          curr_d1 + VIA +
                          v_costTable[v_edges[grid].usage + v_edges[grid].red +
                                      (int)(L * v_edges[grid].last_usage)];
                    }

                    if (d1[tmpY][curX] > tmp && tmp < return_dist) {

                      data[(curY + 1) * xGrid + curX].lock();
                      if (d1[tmpY][curX] > tmp &&
                          tmp < return_dist) // top neighbor been put into heap1
                                             // but needs update
                      {
                        d1[tmpY][curX]       = tmp;
                        parentX1[tmpY][curX] = curX;
                        parentY1[tmpY][curX] = curY;
                        HV[tmpY][curX]       = TRUE;
                        ctx.push(pq_grid(&(d1[tmpY][curX]), tmp));
                      }
                      data[(curY + 1) * xGrid + curX].unlock();
                    }
                  }
                  hyperV[curY][curX] = tmpV;
                }
              },
              // galois::wl<galois::worklists::ParaMeter<>>(),
              // galois::wl<PSChunk>(),
              galois::wl<OBIM>(RequestIndexer)
              // galois::chunk_size<MAZE_CHUNK_SIZE>()
              // galois::parallel_break(),
              // galois::steal(),
              // galois::loopname("fine_grain")
          );

          // timer_foreach.stop();

          for (local_vec::iterator ii = v2.begin(); ii != v2.end(); ii++)
            pop_heap2[*ii] = FALSE;

          crossX = return_ind1 % xGrid;
          crossY = return_ind1 / xGrid;

          cnt      = 0;
          int curX = crossX;
          int curY = crossY;
          int tmpX, tmpY;
          acc_cnt++;
          acc_dist += return_dist;
          max_dist =
              (max_dist >= return_dist.load()) ? max_dist : return_dist.load();

          while (d1[curY][curX] != 0) // loop until reach subtree1
          {
            hypered = FALSE;
            if (cnt != 0) {
              if (curX != tmpX && hyperH[curY][curX]) {
                curX    = 2 * curX - tmpX;
                hypered = TRUE;
              }
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

          // change the tree structure according to the new routing for the tree
          // edge find E1 and E2, and the endpoints of the edges they are on
          E1x = gridsX[0];
          E1y = gridsY[0];
          E2x = gridsX[cnt_n1n2 - 1];
          E2y = gridsY[cnt_n1n2 - 1];

          edge_n1n2 = edgeID;

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

            if (endpt1 == n1 || endpt2 == n1) // E1 is on (n1, A1) or (n1, A2)
            {
              // if E1 is on (n1, A2), switch A1 and A2 so that E1 is always on
              // (n1, A1)
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

            if (endpt1 == n2 || endpt2 == n2) // E2 is on (n2, B1) or (n2, B2)
            {
              // if E2 is on (n2, B2), switch B1 and B2 so that E2 is always on
              // (n2, B1)
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
          // std::cout << " adjsut tree" << std::endl;
          // timer_adjusttree.stop();

          // update edge usage

          // timer_updateusage.start();
          acc_length += cnt_n1n2;
          max_length = (max_length >= cnt_n1n2) ? max_length : cnt_n1n2;
          for (i = 0; i < cnt_n1n2 - 1; i++) {
            if (gridsX[i] == gridsX[i + 1]) // a vertical edge
            {
              min_y = min(gridsY[i], gridsY[i + 1]);
              // v_edges[min_y*xGrid+gridsX[i]].usage += 1;
              // galois::atomicAdd(v_edges[min_y*xGrid+gridsX[i]].usage, (short
              // unsigned)1);
              v_edges[min_y * xGrid + gridsX[i]].usage.fetch_add(
                  (short int)1, std::memory_order_relaxed);
            } else /// if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
              min_x = min(gridsX[i], gridsX[i + 1]);
              // h_edges[gridsY[i]*(xGrid-1)+min_x].usage += 1;
              // galois::atomicAdd(h_edges[gridsY[i]*(xGrid-1)+min_x].usage,
              // (short unsigned)1);
              h_edges[gridsY[i] * (xGrid - 1) + min_x].usage.fetch_add(
                  (short int)1, std::memory_order_relaxed);
            }
          }
          // timer_updateusage.stop();
          // timer_checkroute2dtree.start();
          if (checkRoute2DTree(netID)) {
            reInitTree(netID);
            return;
          }
          // timer_checkroute2dtree.stop();
        } // congested route, if(enter)
        timer_finegrain.stop();
      } // only route the non-degraded edges (len>0)
    }   // iterate on edges of a net
  }

  printf("total ripups: %d max ripups: %d\n", total_ripups.reduce(),
         max_ripups.reduce());
  if (acc_cnt != 0) {
    cout << " max_dist " << max_dist << " max_length " << max_length << endl;
    cout << " avg dist " << acc_dist / acc_cnt << " avg length "
         << (float)acc_length / acc_cnt << " acc_cnt: " << acc_cnt << endl;
    round_avg_dist   = acc_dist / acc_cnt;
    round_avg_length = acc_length / acc_cnt;
  }
  free(h_costTable);
  free(v_costTable);
}

void mazeRouteMSMD_finegrain_doall(int iter, int expand, float costHeight,
                                   int ripup_threshold, int mazeedge_Threshold,
                                   Bool Ordering, int cost_type) {
  // LOCK = 0;
  galois::StatTimer timer_finegrain("fine grain function", "fine grain maze");

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
            (float)costHeight / slope * (i - hCapacity);
    }
    forange = 40 * vCapacity;
    for (int i = 0; i < forange; i++) {
      if (i < vCapacity - 1)
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i - 1) * LOGIS_COF) + 1) + 1;
      else
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i - 1) * LOGIS_COF) + 1) + 1 +
            (float)costHeight / slope * (i - vCapacity);
    }
  } else {

    for (int i = 0; i < forange; i++) {
      if (i < hCapacity)
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) + 1;
      else
        h_costTable[i] =
            costHeight / (exp((float)(hCapacity - i) * LOGIS_COF) + 1) + 1 +
            (float)costHeight / slope * (i - hCapacity);
    }
    forange = 40 * vCapacity;
    for (int i = 0; i < forange; i++) {
      if (i < vCapacity)
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i) * LOGIS_COF) + 1) + 1;
      else
        v_costTable[i] =
            costHeight / (exp((float)(vCapacity - i) * LOGIS_COF) + 1) + 1 +
            (float)costHeight / slope * (i - vCapacity);
    }
  }

  /*forange = yGrid*xGrid;
  for(i=0; i<forange; i++)
  {
      pop_heap2[i] = FALSE;
  } //Michael*/

  galois::LargeArray<galois::substrate::SimpleLock> data;
  data.allocateInterleaved(xGrid * yGrid);

  if (Ordering) {
    StNetOrder();
    // printf("order?\n");
  }

  THREAD_LOCAL_STORAGE thread_local_storage{};
  // for(nidRPC=0; nidRPC<numValidNets; nidRPC++)//parallelize
  PerThread_PQ perthread_pq;
  PerThread_Vec perthread_vec;
  PRINT = 0;
  galois::GAccumulator<int> total_ripups;
  galois::GReduceMax<int> max_ripups;
  total_ripups.reset();
  max_ripups.reset();

  galois::StatTimer timer_newripupcheck("ripup", "fine grain maze");
  galois::StatTimer timer_setupheap("setup heap", "fine grain maze");
  galois::StatTimer timer_traceback("trace back", "fine grain maze");
  galois::StatTimer timer_adjusttree("adjust tree", "fine grain maze");
  galois::StatTimer timer_updateusage("update usage", "fine grain maze");
  galois::StatTimer timer_checkroute2dtree("checkroute2dtree",
                                           "fine grain maze");
  galois::StatTimer timer_init("init", "fine grain maze");
  galois::StatTimer timer_foreach("foreach", "fine grain maze");
  for (int nidRPC = 0; nidRPC < numValidNets; nidRPC++) {

    int netID;

    // maze routing for multi-source, multi-destination
    Bool hypered, enter;
    int i, j, deg, edgeID, n1, n2, n1x, n1y, n2x, n2y, ymin, ymax, xmin, xmax,
        crossX, crossY, tmpi, min_x, min_y, num_edges;
    int regionX1, regionX2, regionY1, regionY2;
    int tmpind, gridsX[XRANGE], gridsY[YRANGE], tmp_gridsX[XRANGE],
        tmp_gridsY[YRANGE];
    int endpt1, endpt2, A1, A2, B1, B2, C1, C2, D1, D2, cnt, cnt_n1n2;
    int edge_n1n2, edge_n1A1, edge_n1A2, edge_n1C1, edge_n1C2, edge_A1A2,
        edge_C1C2;
    int edge_n2B1, edge_n2B2, edge_n2D1, edge_n2D2, edge_B1B2, edge_D1D2;
    int E1x, E1y, E2x, E2y;
    int origENG, edgeREC;

    TreeEdge *treeedges, *treeedge;
    TreeNode* treenodes;

    bool* pop_heap2 = thread_local_storage.pop_heap2;

    float** d1    = thread_local_storage.d1_p;
    bool** HV     = thread_local_storage.HV_p;
    bool** hyperV = thread_local_storage.hyperV_p;
    bool** hyperH = thread_local_storage.hyperH_p;

    short** parentX1 = thread_local_storage.parentX1_p;
    short** parentX3 = thread_local_storage.parentX3_p;
    short** parentY1 = thread_local_storage.parentY1_p;
    short** parentY3 = thread_local_storage.parentY3_p;

    int** corrEdge = thread_local_storage.corrEdge_p;

    OrderNetEdge* netEO = thread_local_storage.netEO_p;

    bool** inRegion = thread_local_storage.inRegion_p;

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
    // parentY1[153][134], parentX3[153][134]); printf("what is happening?\n");

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
        timer_newripupcheck.start();
        enter = newRipupCheck(treeedge, ripup_threshold, netID, edgeID);
        timer_newripupcheck.stop();

        // ripup the routing for the edge
        timer_finegrain.start();
        if (enter) {
          // if(netID == 2 && edgeID == 26)
          //    printf("netID %d edgeID %d src %d %d dst %d %d\n", netID,
          //    edgeID, n1x, n1y, n2x, n2y);
          // pre_length = treeedge->route.routelen;
          /*for(int i = 0; i < pre_length; i++)
          {
              pre_gridsY[i] = treeedge->route.gridsY[i];
              pre_gridsX[i] = treeedge->route.gridsX[i];
              //printf("i %d x %d y %d\n", i, pre_gridsX[i], pre_gridsY[i]);
          }*/
          timer_init.start();
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

          int enlarge = min(
              origENG, (iter / 6 + 3) *
                           treeedge->route
                               .routelen); // michael, this was global variable
          regionX1 = max(0, xmin - enlarge);
          regionX2 = min(xGrid - 1, xmax + enlarge);
          regionY1 = max(0, ymin - enlarge);
          regionY2 = min(yGrid - 1, ymax + enlarge);
          // std::cout << "region size" << regionWidth << ", " << regionHeight
          // << std::endl;
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
              HV[i][j] = FALSE;
            }
          }
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
          timer_setupheap.start();
          setupHeap(netID, edgeID, pq1, v2, regionX1, regionX2, regionY1,
                    regionY2, d1, corrEdge, inRegion);
          timer_setupheap.stop();
          // TODO: use std priority queue
          // while loop to find shortest path
          /*ind1 = (pq1.top().d1_p - &d1[0][0]);
          curX = ind1%xGrid;
          curY = ind1/xGrid;
          printf("src size: %d dst size: %d\n", pq1.size(), v2.size());*/
          for (local_vec::iterator ii = v2.begin(); ii != v2.end(); ii++) {
            pop_heap2[*ii] = TRUE;
          }
          std::atomic<int> return_ind1;
          std::atomic<float> return_dist;
          return_dist = (float)BIG_INT;

          timer_init.stop();
          timer_foreach.start();

          galois::InsertBag<pq_grid> wls[2];
          galois::InsertBag<pq_grid>* next;
          galois::InsertBag<pq_grid>* cur;

          cur  = &wls[0];
          next = &wls[1];
          std::atomic<int> bagsize;
          bagsize = 0;

          while (!pq1.empty()) {
            auto tmp = pq1.top();
            pq1.pop();
            cur->push(tmp);
            // bagsize++;
          }

          while (!cur->empty()) {
            // std::cout << "bag size: " << bagsize.load() << " unique items: "
            // << dynBS.count() << std::endl; bagsize = 0; dynBS.reset();
            // galois::for_each(galois::iterate(*cur),
            //[&] (const auto& top, auto& ctx)
            galois::do_all(
                galois::iterate(*cur),
                [&](const auto& top)

                /*galois::for_each(galois::iterate(pq1),
                [&] (const auto& top, auto& ctx)*/
                // while( pop_heap2[ind1]==FALSE) // stop until the grid
                // position been popped out from both heap1 and heap2
                {
                  // relax all the adjacent grids within the enlarged region for
                  // source subtree

                  int ind1 = top.d1_p - &d1[0][0];

                  //
                  int curX = ind1 % xGrid;
                  int curY = ind1 / xGrid;
                  int grid = curY * xGrid + curX;
                  // std::cout << " d1: " << d1[curY][curX] << std::endl;
                  float d1_push = top.d1_push; // d1[curY][curX];
                  float curr_d1 = d1[curY][curX];
                  // std::cout << "d1_push: " << d1_push << " d1: " <<
                  // d1[curY][curX] << std::endl; if(netID == 2 && edgeID == 26)
                  // printf("netID: %d edgeID:%d curX curY %d %d, d1_push: %f,
                  // curr_d1: %f\n", netID, edgeID, curX, curY, d1_push,
                  // curr_d1);

                  if (d1_push > return_dist + OBIM_delta) {
                    // printf("netID: %d early break\n", netID);
                    // if(netID == 2 && edgeID == 26)
                    //    printf("break! curX curY %d %d, d1_push: %f, curr_d1:
                    //    %f return_d: %f\n", curX, curY, d1_push, curr_d1,
                    //    return_dist.load());
                    // ctx.breakLoop();
                  }
                  // galois::runtime::acquire(&data[grid],
                  // galois::MethodFlag::WRITE);
                  if (d1_push == curr_d1 && d1_push < return_dist.load()) {
                    if (pop_heap2[ind1] != false) {
                      // if(netID == 2 && edgeID == 26)
                      //    printf("reach! curX curY %d %d, d1_push: %f,
                      //    curr_d1: %f return_d: %f\n", curX, curY, d1_push,
                      //    curr_d1, return_dist.load());
                      return_ind1.store(ind1);
                      return_dist.store(d1_push);
                    }

                    /*grid = curY*xGrid + curX - 1;
                    if(curX>regionX1)
                        galois::runtime::acquire(&data[grid],
                    galois::MethodFlag::WRITE);

                    grid = curY*xGrid + curX + 1;
                    if(curX<regionX2)
                        galois::runtime::acquire(&data[grid],
                    galois::MethodFlag::WRITE);

                    grid = (curY - 1)*xGrid + curX;
                    if(curY>regionY1)
                        galois::runtime::acquire(&data[grid],
                    galois::MethodFlag::WRITE);

                    grid = (curY + 1)*xGrid + curX;
                    if(curY<regionY2)
                        galois::runtime::acquire(&data[grid],
                    galois::MethodFlag::WRITE);*/

                    int preX = curX, preY = curY;
                    if (curr_d1 != 0) {
                      if (HV[curY][curX]) {
                        preX = parentX1[curY][curX];
                        preY = parentY1[curY][curX];
                      } else {
                        preX = parentX3[curY][curX];
                        preY = parentY3[curY][curX];
                      }
                    }
                    // printf("pop curY: %d curX: %d d1: %f preX: %d preY: %d
                    // hyperH: %d hyperV: %d HV: %d return_dist: %f\n",
                    //    curY, curX, curr_d1, preX, preY, hyperH[curY][curX],
                    //    hyperV[curY][curX], HV[curY][curX],
                    //    return_dist.load());
                    float tmp = 0.f, tmp_cost = 0.f;
                    int tmp_grid = 0;
                    int tmpX = 0, tmpY = 0;
                    // left
                    bool tmpH = false;
                    bool tmpV = false;

                    // if(curX>regionX1)
                    //    data[curY*xGrid+curX-1].lock();

                    // data[curY*(xGrid-1)+curX].lock();

                    if (curX > regionX1) {
                      grid = curY * (xGrid - 1) + curX - 1;

                      // printf("grid: %d %d usage: %d red:%d last:%d sum%f
                      // %d\n",
                      //    grid%xGrid, grid/xGrid, h_edges[grid].usage.load(),
                      //    h_edges[grid].red, h_edges[grid].last_usage, L ,
                      //    h_edges[grid].usage.load() + h_edges[grid].red +
                      //    (int)(L*h_edges[grid].last_usage));
                      if ((preY == curY) || (curr_d1 == 0)) {
                        tmp = curr_d1 +
                              h_costTable[h_edges[grid].usage +
                                          h_edges[grid].red +
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
                            // hyperH[curY][curX] = TRUE; //Michael
                            tmpH = true;
                          }
                        }
                        tmp = curr_d1 + VIA +
                              h_costTable[h_edges[grid].usage +
                                          h_edges[grid].red +
                                          (int)(L * h_edges[grid].last_usage)];
                      }
                      tmpX = curX - 1; // the left neighbor
                      if (d1[curY][tmpX] > tmp && tmp < return_dist) {
                        data[curY * xGrid + curX - 1].lock();
                        if (d1[curY][tmpX] > tmp &&
                            tmp < return_dist) // left neighbor been put into
                                               // heap1 but needs update
                        {
                          d1[curY][tmpX]       = tmp;
                          parentX3[curY][tmpX] = curX;
                          parentY3[curY][tmpX] = curY;
                          HV[curY][tmpX]       = FALSE;
                          // pq1.push({&(d1[curY][tmpX]), tmp});
                          // pq_grid grid_push = {&(d1[curY][tmpX]), tmp};

                          // next->push(&(d1[curY][tmpX]));
                          next->push(pq_grid(&(d1[curY][tmpX]), tmp));
                          // bagsize.fetch_add(1, std::memory_order_relaxed);

                          // printf("left push Y: %d X: %d tmp: %f HV: false
                          // hyperH: %d\n", curY, tmpX, tmp, true);
                        }
                        data[curY * xGrid + curX - 1].unlock();
                      }
                    }
                    // right

                    if (curX < regionX2) {
                      // data[curY*xGrid+curX+1].lock();
                      grid = curY * (xGrid - 1) + curX;
                      // printf("grid: %d %d usage: %d red:%d last:%d L:%f
                      // sum:%d\n",grid%xGrid, grid/xGrid,
                      // h_edges[grid].usage.load(), h_edges[grid].red,
                      // h_edges[grid].last_usage, L ,
                      // h_edges[grid].usage.load()
                      // + h_edges[grid].red +
                      // (int)(L*h_edges[grid].last_usage));
                      if ((preY == curY) || (curr_d1 == 0)) {
                        tmp = curr_d1 +
                              h_costTable[h_edges[grid].usage +
                                          h_edges[grid].red +
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
                            // hyperH[curY][curX] = TRUE;
                            tmpH = true;
                          }
                        }
                        tmp = curr_d1 + VIA +
                              h_costTable[h_edges[grid].usage +
                                          h_edges[grid].red +
                                          (int)(L * h_edges[grid].last_usage)];
                      }
                      tmpX = curX + 1; // the right neighbor

                      if (d1[curY][tmpX] > tmp && tmp < return_dist) {
                        data[curY * xGrid + curX + 1].lock();
                        if (d1[curY][tmpX] > tmp &&
                            tmp < return_dist) // right neighbor been put into
                                               // heap1 but needs update
                        {
                          d1[curY][tmpX]       = tmp;
                          parentX3[curY][tmpX] = curX;
                          parentY3[curY][tmpX] = curY;
                          HV[curY][tmpX]       = FALSE;

                          // next->push(&(d1[curY][tmpX]));
                          next->push(pq_grid(&(d1[curY][tmpX]), tmp));
                          // bagsize.fetch_add(1, std::memory_order_relaxed);

                          // printf("right push Y: %d X: %d tmp: %f HV: false
                          // hyperH: %d\n", curY, tmpX, tmp, true);
                        }
                        data[curY * xGrid + curX + 1].unlock();
                      }
                    }
                    // data[curY*(xGrid-1)+curX].lock();
                    hyperH[curY][curX] = tmpH;

                    // bottom

                    if (curY > regionY1) {
                      grid = (curY - 1) * xGrid + curX;
                      // printf("grid: %d %d usage: %d red:%d last:%d sum%f
                      // %d\n",
                      //    grid%xGrid, grid/xGrid, v_edges[grid].usage.load(),
                      //    v_edges[grid].red, v_edges[grid].last_usage, L ,
                      //    v_edges[grid].usage.load() + v_edges[grid].red +
                      //    (int)(L*v_edges[grid].last_usage));
                      if ((preX == curX) || (curr_d1 == 0)) {
                        tmp = curr_d1 +
                              v_costTable[v_edges[grid].usage +
                                          v_edges[grid].red +
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
                            // hyperV[curY][curX] = TRUE;
                            tmpV = true;
                          }
                        }
                        tmp = curr_d1 + VIA +
                              v_costTable[v_edges[grid].usage +
                                          v_edges[grid].red +
                                          (int)(L * v_edges[grid].last_usage)];
                      }
                      tmpY = curY - 1; // the bottom neighbor

                      if (d1[tmpY][curX] > tmp && tmp < return_dist) {
                        data[(curY - 1) * xGrid + curX].lock();
                        if (d1[tmpY][curX] > tmp &&
                            tmp < return_dist) // bottom neighbor been put into
                                               // heap1 but needs update
                        {
                          d1[tmpY][curX]       = tmp;
                          parentX1[tmpY][curX] = curX;
                          parentY1[tmpY][curX] = curY;
                          HV[tmpY][curX]       = TRUE;

                          // next->push(&(d1[tmpY][curX]));
                          next->push(pq_grid(&(d1[tmpY][curX]), tmp));
                          // bagsize.fetch_add(1, std::memory_order_relaxed);
                        }
                        data[(curY - 1) * xGrid + curX].unlock();
                      }
                    }
                    // top
                    if (curY < regionY2) {

                      grid = curY * xGrid + curX;
                      // printf("grid: %d %d usage: %d red:%d last:%d sum%f
                      // %d\n",
                      //    grid%xGrid, grid/xGrid, v_edges[grid].usage.load(),
                      //    v_edges[grid].red, v_edges[grid].last_usage, L ,
                      //    v_edges[grid].usage.load() + v_edges[grid].red +
                      //    (int)(L*v_edges[grid].last_usage));
                      if ((preX == curX) || (curr_d1 == 0)) {
                        tmp = curr_d1 +
                              v_costTable[v_edges[grid].usage +
                                          v_edges[grid].red +
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
                            // hyperV[curY][curX] = TRUE;
                            tmpV = true;
                          }
                        }
                        tmp = curr_d1 + VIA +
                              v_costTable[v_edges[grid].usage +
                                          v_edges[grid].red +
                                          (int)(L * v_edges[grid].last_usage)];
                      }
                      tmpY = curY + 1; // the top neighbor

                      if (d1[tmpY][curX] > tmp && tmp < return_dist) {
                        data[(curY + 1) * xGrid + curX].lock();
                        if (d1[tmpY][curX] > tmp &&
                            tmp < return_dist) // top neighbor been put into
                                               // heap1 but needs update
                        {

                          d1[tmpY][curX]       = tmp;
                          parentX1[tmpY][curX] = curX;
                          parentY1[tmpY][curX] = curY;
                          HV[tmpY][curX]       = TRUE;

                          // next->push(&(d1[tmpY][curX]));
                          next->push(pq_grid(&(d1[tmpY][curX]), tmp));
                          // bagsize.fetch_add(1, std::memory_order_relaxed);
                        }
                        data[(curY + 1) * xGrid + curX].unlock();
                      }
                    }
                    hyperV[curY][curX] = tmpV;
                  }
                },
                // galois::wl<galois::worklists::ParaMeter<>>(),
                // galois::wl<PSChunk>(),
                // galois::wl<OBIM>(RequestIndexer),
                galois::chunk_size<4>(),
                // galois::parallel_break(),
                // galois::steal(),
                galois::loopname("fine_grain"));
            std::swap(cur, next);
            next->clear();

          } // do all while curr is not empty

          timer_foreach.stop();

          for (local_vec::iterator ii = v2.begin(); ii != v2.end(); ii++)
            pop_heap2[*ii] = FALSE;

          crossX = return_ind1 % xGrid;
          crossY = return_ind1 / xGrid;

          cnt      = 0;
          int curX = crossX;
          int curY = crossY;
          int tmpX, tmpY;
          // if(netID == 2 && edgeID == 26)
          //    printf("crossX %d crossY %d return_d: %f\n", crossX, crossY,
          //    return_dist.load());
          timer_traceback.start();
          while (d1[curY][curX] != 0) // loop until reach subtree1
          {
            // if(cnt > 1000)
            //    printf("Y: %d X: %d hyperH: %d hyperV: %d HV: %d d1: %f\n",
            //    curY, curX, hyperH[curY][curX], hyperV[curY][curX],
            //    HV[curY][curX], d1[curY][curX]);

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

          // change the tree structure according to the new routing for the tree
          // edge find E1 and E2, and the endpoints of the edges they are on
          E1x = gridsX[0];
          E1y = gridsY[0];
          E2x = gridsX[cnt_n1n2 - 1];
          E2y = gridsY[cnt_n1n2 - 1];

          edge_n1n2 = edgeID;

          timer_traceback.stop();

          // if(netID == 14628)
          //    printf("netID %d edgeID %d src %d %d dst %d %d routelen: %d\n",
          //    netID, edgeID, E1x, E1y, E2x, E2y, cnt_n1n2);
          // (1) consider subtree1
          timer_adjusttree.start();
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

            if (endpt1 == n1 || endpt2 == n1) // E1 is on (n1, A1) or (n1, A2)
            {
              // if E1 is on (n1, A2), switch A1 and A2 so that E1 is always on
              // (n1, A1)
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

            if (endpt1 == n2 || endpt2 == n2) // E2 is on (n2, B1) or (n2, B2)
            {
              // if E2 is on (n2, B2), switch B1 and B2 so that E2 is always on
              // (n2, B1)
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
          // std::cout << " adjsut tree" << std::endl;
          timer_adjusttree.stop();

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
          int)1, std::memory_order_relaxed);
                  //if(v_edges[min_y*xGrid+pre_gridsX[i]].usage < 0) printf("V
          negative! %d \n", i);
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
          int)1, std::memory_order_relaxed);
                  //if(h_edges[pre_gridsY[i]*(xGrid-1)+min_x].usage < 0)
          printf("H negative! %d \n", i);
              }
          }*/
          timer_updateusage.start();
          for (i = 0; i < cnt_n1n2 - 1; i++) {
            if (gridsX[i] == gridsX[i + 1]) // a vertical edge
            {
              min_y = min(gridsY[i], gridsY[i + 1]);
              // v_edges[min_y*xGrid+gridsX[i]].usage += 1;
              // galois::atomicAdd(v_edges[min_y*xGrid+gridsX[i]].usage, (short
              // unsigned)1);
              v_edges[min_y * xGrid + gridsX[i]].usage.fetch_add(
                  (short int)1, std::memory_order_relaxed);
            } else /// if(gridsY[i]==gridsY[i+1])// a horizontal edge
            {
              min_x = min(gridsX[i], gridsX[i + 1]);
              // h_edges[gridsY[i]*(xGrid-1)+min_x].usage += 1;
              // galois::atomicAdd(h_edges[gridsY[i]*(xGrid-1)+min_x].usage,
              // (short unsigned)1);
              h_edges[gridsY[i] * (xGrid - 1) + min_x].usage.fetch_add(
                  (short int)1, std::memory_order_relaxed);
            }
          }
          timer_updateusage.stop();
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
          // printf("netID %d edgeID %d src %d %d dst %d %d routelen: %d\n",
          // netID, edgeID, n1x, n1y, n2x, n2y, cnt_n1n2);
          timer_checkroute2dtree.start();
          if (checkRoute2DTree(netID)) {
            reInitTree(netID);
            return;
          }
          timer_checkroute2dtree.stop();
        } // congested route, if(enter)
        timer_finegrain.stop();
      } // only route the non-degraded edges (len>0)
    }   // iterate on edges of a net
  }

  printf("total ripups: %d max ripups: %d\n", total_ripups.reduce(),
         max_ripups.reduce());
  //}, "mazeroute vtune function");
  free(h_costTable);
  free(v_costTable);
}
