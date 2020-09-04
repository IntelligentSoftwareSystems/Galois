#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
#include "llvm/Support/CommandLine.h"

#include "BoilerPlate.h"
#include "Lonestar/BFS_SSSP.h"

#include "DataType.h"
#include "flute.h"
#include "DataProc.h"
#include "RSMT.h"
#include "maze.h"
#include "RipUp.h"
#include "utility.h"
#include "route.h"
#include "maze3D.h"
#include "maze_finegrain.h"
#include "maze_finegrain_lateupdate.h"
#include "maze_lock.h"

static const char* name = "SPRoute";

static const char* desc =
    "A Scalable Parallel global router with a hybrid parallel algorithm which "
    "combines net-level parallelism and fine-grain parallelism";

static const char* url = "SPRoute";

namespace cll = llvm::cl;
static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);

static cll::opt<std::string> outfile("o", cll::desc("output file (optional)"),
                                     cll::init(""));

static cll::opt<std::string>
    fluteDir("flute",
             cll::desc("directory of POWV9.dat and POST9.dat (REQUIRED)"),
             cll::Required);

//! Flag that forces user to be aware that they should be passing in a
//! ISPD2008 graph.
static cll::opt<bool> ISPD2008Graph(
    "ISPD2008Graph",
    cll::desc("Specify that the input graph is a ISPD2008 graph format"),
    cll::init(false));

int main(int argc, char** argv) {
  //    char benchFile[FILESTRLEN];
  clock_t t1, t2, t3;
  float gen_brk_Time, reading_Time;
  int enlarge, ripup_threshold;
  int i;
  int ESTEP1, CSTEP1, thStep1;
  int ESTEP2, CSTEP2, thStep2;
  int ESTEP3, CSTEP3, tUsage;
  int Ripvalue, LVIter, cost_step;
  int maxOverflow, past_cong, last_cong, finallength, numVia, ripupTH3D, newTH,
      healingTrigger;
  int minofl, minoflrnd = 0, mazeRound, upType, cost_type, bmfl, bwcnt;
  bool goingLV, noADJ, needOUTPUT;

  needOUTPUT = false;

  /*string outFile;
  for(int i = 1; i < argc; i++) {
      string tmp(argv[i]);
      if(tmp == "-t")
          numThreads = atoi(argv[i+1]);
      else if(tmp == "-o") {
          outFile = string(argv[i+1]);
          needOUTPUT = true;
      }
      else if(tmp == "-h" || tmp == "--help") {
          printf("Usage: ./SPRoute  <input> -o <output> -t <nthreads> \n");
          exit(1);
      }
  }*/

  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url, &inputFile);

  galois::preAlloc(numThreads * 2);

  if (!ISPD2008Graph) {
    GALOIS_DIE("This application requires a ISPD2008 graph input;"
               " please use the -ISPD2008Graph flag "
               " to indicate the input is a ISPD2008 graph format.");
  }

  if (outfile != "") {
    needOUTPUT = true;
  }
  LB = 0.9;
  UB = 1.3;

  SLOPE     = 5;
  THRESH_M  = 20;
  ENLARGE   = 15; // 5
  ESTEP1    = 10; // 10
  ESTEP2    = 5;  // 5
  ESTEP3    = 5;  // 5
  CSTEP1    = 2;  // 5
  CSTEP2    = 2;  // 3
  CSTEP3    = 5;  // 15
  COSHEIGHT = 4;
  L         = 0;
  VIA       = 2;
  Ripvalue  = -1;
  ripupTH3D = 10;
  goingLV   = TRUE;
  noADJ     = FALSE;
  thStep1   = 10;
  thStep2   = 4;
  LVIter    = 3;
  mazeRound = 500;
  bmfl      = BIG_INT;
  minofl    = BIG_INT;

  // galois::substrate::PerThreadStorage<THREAD_LOCAL_STORAGE>
  // thread_local_storage; galois::setActiveThreads(numThreads);
  /* galois::on_each(
           [&] (const unsigned tid, const unsigned numT)
           {
               printf("threadid: %d %d\n", tid, numT);
           }
           );
*/
  cout << " nthreads: " << numThreads << endl;

  int finegrain     = false;
  int thread_choice = 0;
  // int thread_steps[6] = {28,14,8,4,1};
  // int thread_livelock_limit[6] = {1,1,1,1,1};
  bool extrarun       = false;
  int thread_livelock = 0;

  if (1) {
    t1 = clock();
    printf("\nReading %s ...\n", inputFile.c_str());
    readFile(inputFile.c_str());
    printf("\nReading Lookup Table ...\n");
    readLUT(fluteDir.c_str());
    printf("\nDone reading table\n\n");

    t2           = clock();
    reading_Time = (float)(t2 - t1) / CLOCKS_PER_SEC;
    printf("Reading Time: %f sec\n", reading_Time);

    // call FLUTE to generate RSMT and break the nets into segments (2-pin nets)

    VIA = 2;
    // viacost = VIA;
    viacost = 0;
    gen_brk_RSMT(FALSE, FALSE, FALSE, FALSE, noADJ);
    printf("first L\n");
    routeLAll(TRUE);
    gen_brk_RSMT(TRUE, TRUE, TRUE, FALSE, noADJ);
    getOverflow2D(&maxOverflow);
    printf("second L\n");
    newrouteLAll(FALSE, TRUE);
    getOverflow2D(&maxOverflow);
    spiralRouteAll();
    newrouteZAll(10);
    printf("first Z\n");
    past_cong = getOverflow2D(&maxOverflow);

    convertToMazeroute();

    enlarge        = 10;
    newTH          = 10;
    healingTrigger = 0;
    stopDEC        = 0;
    upType         = 1;

    // iniBDE();

    costheight = COSHEIGHT;

    if (maxOverflow > 700) {
      costheight = 8;
      LOGIS_COF  = 1.33;
      VIA        = 0;
      THRESH_M   = 0;
      CSTEP1     = 30;
      slope      = BIG_INT;
    }

    for (i = 0; i < LVIter; i++) {

      LOGIS_COF = max(2.0 / (1 + log(maxOverflow)), LOGIS_COF);
      LOGIS_COF = 2.0 / (1 + log(maxOverflow));
      printf("LV routing round %d, enlarge %d \n", i, enlarge);
      routeLVAll(newTH, enlarge);

      past_cong = getOverflow2Dmaze(&maxOverflow, &tUsage);

      enlarge += 5;
      newTH -= 5;
      if (newTH < 1) {
        newTH = 1;
      }
    }

    //	past_cong = getOverflow2Dmaze( &maxOverflow);

    t3           = clock();
    reading_Time = (float)(t3 - t2) / CLOCKS_PER_SEC;
    printf("LV Time: %f sec\n", reading_Time);
    InitEstUsage();

    i               = 1;
    costheight      = COSHEIGHT;
    enlarge         = ENLARGE;
    ripup_threshold = Ripvalue;

    minofl  = totalOverflow;
    stopDEC = FALSE;

    slope     = 20;
    L         = 1;
    cost_type = 1;

    InitLastUsage(upType);

    // OrderNetEdge* netEO = (OrderNetEdge*)calloc(2000, sizeof(OrderNetEdge));

    PRINT_HEAT = 0;
    // checkUsageCorrectness();
    galois::StatTimer roundtimer("round");
    unsigned long oldtime = 0;
    round_avg_dist        = 0;
    round_avg_length      = 0;
    while (totalOverflow > 0) {

      if (THRESH_M > 15) {
        THRESH_M -= thStep1;
      } else if (THRESH_M >= 2) {
        THRESH_M -= thStep2;
      } else {
        THRESH_M = 0;
      }
      if (THRESH_M <= 0) {
        THRESH_M = 0;
      }
      // std::cout << "totalOverflow : " << totalOverflow << " enlarge: " <<
      // enlarge << std::endl;
      if (totalOverflow > 2000) {
        enlarge += ESTEP1; // ENLARGE+(i-1)*ESTEP;
        cost_step = CSTEP1;
        updateCongestionHistory(upType);

      } else if (totalOverflow < 500) {

        cost_step = CSTEP3;
        enlarge += ESTEP3;
        ripup_threshold = -1;
        updateCongestionHistory(upType);
      } else {
        cost_step = CSTEP2;
        enlarge += ESTEP2;
        updateCongestionHistory(upType);
      }

      if (totalOverflow > 15000 && maxOverflow > 400) {
        enlarge = max(xGrid, yGrid) /
                  30; // This is the key!!!! to enlarge routing area!!!!
        // enlarge = max(xGrid,yGrid) / 10;
        slope = BIG_INT;
        // slope = 20;
        if (i == 5) {
          VIA             = 0;
          LOGIS_COF       = 1.33;
          ripup_threshold = -1;
          //	cost_type = 3;

        } else if (i > 6) {
          if (i % 2 == 0) {
            LOGIS_COF += 0.5;
          }
          if (i > 20) {
            break;
          }
        }
        if (i > 10) {
          cost_type       = 1;
          ripup_threshold = 0;
        }
      }

      int maxGrid = max(xGrid + 1, yGrid + 1);
      enlarge     = min(enlarge, maxGrid / 2);
      // std::cout << "costheight : " << costheight << " enlarge: " << enlarge
      // << std::endl;
      costheight += cost_step;
      // std::cout << "costheight : " << costheight << " enlarge: " << enlarge
      // << std::endl;
      mazeedge_Threshold = THRESH_M;

      if (upType == 3) {
        LOGIS_COF = max(2.0 / (1 + log(maxOverflow + max_adj)), LOGIS_COF);
      } else {
        LOGIS_COF = max(2.0 / (1 + log(maxOverflow)), LOGIS_COF);
      }

      if (i == 8) {
        L      = 0;
        upType = 2;
        InitLastUsage(upType);
      }

      if (maxOverflow == 1) {
        // L = 0;
        ripup_threshold = -1;
        slope           = 5;
      }

      if (maxOverflow > 300 && past_cong > 15000) {
        L = 0;
      }
      // checkUsageCorrectness();

      // getOverflow2Dmaze(&maxOverflow , & tUsage);

      printf("iteration %d, enlarge %d, costheight %d, threshold %d via cost "
             "%d \nlog_coef %f, healingTrigger %d cost_step %d slope %d L %f "
             "cost_type %d OBIM delta %d\n",
             i, enlarge, costheight, mazeedge_Threshold, VIA, LOGIS_COF,
             healingTrigger, cost_step, slope, L, cost_type,
             max(OBIM_delta, (int)(costheight / (2 * slope))));
      // L = 2;
      roundtimer.start();
      round_num = i;
      if (finegrain) {
        printf("finegrain\n");

        mazeRouteMSMD_finegrain_spinlock(i, enlarge, costheight,
                                         ripup_threshold, mazeedge_Threshold,
                                         !(i % 3), cost_type);
      } else {
        mazeRouteMSMD(i, enlarge, costheight, ripup_threshold,
                      mazeedge_Threshold, !(i % 3), cost_type);
      }
      roundtimer.stop();
      cout << "round : " << i << " time(ms): " << roundtimer.get() - oldtime
           << " acc time(ms): " << roundtimer.get() << endl;
      oldtime = roundtimer.get();

      last_cong        = past_cong;
      past_cong        = getOverflow2Dmaze(&maxOverflow, &tUsage);
      int nthreads_tmp = numThreads;
      if (past_cong > last_cong && !extrarun) // Michael
      {
        if (!finegrain && nthreads_tmp != 1) {
          thread_livelock++;
          if (thread_livelock == 1) {
            thread_choice++;
            thread_livelock = 0;
            if (nthreads_tmp < 6) {
              galois::setActiveThreads(4);
              numThreads = 4;
              finegrain  = true;
            } else {
              numThreads = numThreads / 2;
              galois::setActiveThreads(numThreads);
            }
          }
        }
      }
      cout << "nthreads :" << numThreads << endl;
      extrarun = false;

      if (minofl > past_cong) {
        minofl    = past_cong;
        minoflrnd = i;
      }

      if (i == 8) {
        L = 1;
      }

      i++;

      if (past_cong < 200 && i > 30 && upType == 2 && max_adj <= 20) {
        upType  = 4;
        stopDEC = TRUE;
      }

      if (maxOverflow < 150) {
        if (i == 20 && past_cong > 200) {
          printf("Extra Run for hard benchmark\n");
          L       = 0;
          upType  = 3;
          stopDEC = TRUE;
          slope   = 5;
          galois::runtime::profileVtune(
              [&](void) {
                if (finegrain) {
                  printf("finegrain\n");

                  mazeRouteMSMD_finegrain_spinlock(
                      i, enlarge, costheight, ripup_threshold,
                      mazeedge_Threshold, !(i % 3), cost_type);
                } else {
                  mazeRouteMSMD(i, enlarge, costheight, ripup_threshold,
                                mazeedge_Threshold, !(i % 3), cost_type);
                }
              },
              "mazeroute");
          last_cong = past_cong;
          past_cong = getOverflow2Dmaze(&maxOverflow, &tUsage);
          extrarun  = true;

          str_accu(12);
          L       = 1;
          stopDEC = FALSE;
          slope   = 3;
          upType  = 2;
        }
        if (i == 35 && tUsage > 800000) {
          str_accu(25);
          extrarun = true;
        }
        if (i == 50 && tUsage > 800000) {
          str_accu(40);
          extrarun = true;
        }
      }

      if (i > 50) {
        upType = 4;
        if (i > 70) {
          stopDEC = TRUE;
        }
      }

      if (past_cong > 0.7 * last_cong) {
        costheight += CSTEP3;
      }

      if (past_cong >= last_cong) {
        VIA = 0; // is this good?
        healingTrigger++;
      }

      if (past_cong < bmfl) {
        bwcnt = 0;
        if (i > 140 || (i > 80 && past_cong < 20)) {
          copyRS();
          bmfl = past_cong;

          L     = 0;
          slope = BIG_INT;
          // SLOPE = BIG_INT;
          galois::runtime::profileVtune(
              [&](void) {
                if (finegrain) {
                  printf("finegrain\n");

                  mazeRouteMSMD_finegrain_spinlock(
                      i, enlarge, costheight, ripup_threshold,
                      mazeedge_Threshold, !(i % 3), cost_type);
                } else {
                  mazeRouteMSMD(i, enlarge, costheight, ripup_threshold,
                                mazeedge_Threshold, !(i % 3), cost_type);
                }
              },
              "mazeroute");
          last_cong = past_cong;
          past_cong = getOverflow2Dmaze(&maxOverflow, &tUsage);
          extrarun  = true;
          if (past_cong < last_cong) {
            copyRS();
            bmfl = past_cong;
          }
          L     = 1;
          slope = 5;
          // SLOPE = 5;
          if (minofl > past_cong) {
            minofl    = past_cong;
            minoflrnd = i;
          }
          if (bmfl < 72)
            break;
        }
      } else {
        bwcnt++;
      }

      if (bmfl > 10) {
        if (bmfl > 30 && bmfl < 72 && bwcnt > 50) {
          break;
        }
        if (bmfl < 30 && bwcnt > 50) {
          break;
        }
        if (i >= mazeRound) {
          getOverflow2Dmaze(&maxOverflow, &tUsage);
          break;
        }
      }

      if (i >= mazeRound) {
        getOverflow2Dmaze(&maxOverflow, &tUsage);
        break;
      }
    }

    if (minofl > 0) {
      printf("\n\n minimal ofl %d, occuring at round %d\n\n", minofl,
             minoflrnd);
      copyBR();
    }

    freeRR();

    checkUsage();

    printf("maze routing finished\n");

    // t4 = clock();
    // maze_Time = (float)(t4-t3)/CLOCKS_PER_SEC;
    // printf("P3 runtime: %f sec\n", maze_Time);

    printf("Final 2D results: \n");
    getOverflow2Dmaze(&maxOverflow, &tUsage);

    printf("\nLayer Assignment Begins");
    newLA();
    printf("layer assignment finished\n");

    t2           = clock();
    gen_brk_Time = (float)(t2 - t1) / CLOCKS_PER_SEC;
    // printf("2D + Layer Assignment Runtime: %f sec\n", gen_brk_Time);

    costheight = 3;
    viacost    = 1;

    if (gen_brk_Time < 60) {
      ripupTH3D = 15;
    } else if (gen_brk_Time < 120) {
      ripupTH3D = 18;
    } else {
      ripupTH3D = 20;
    }

    if (goingLV && past_cong == 0) {
      printf("Post Processing Begins \n");
      mazeRouteMSMDOrder3D(enlarge, 0, ripupTH3D);

      //	mazeRouteMSMDOrder3D(enlarge, 0, 10 );
      if (gen_brk_Time > 120) {
        mazeRouteMSMDOrder3D(enlarge, 0, 12);
      }
      printf("Post Processsing finished, starting via filling\n");
    }

    fillVIA();
    finallength = getOverflow3D();
    numVia      = threeDVIA();
    checkRoute3D();
    if (needOUTPUT) {
      writeRoute3D(outfile.c_str());
    }

  } // Input ==1

  // t4 = clock();
  // maze_Time = (float)(t4-t1)/CLOCKS_PER_SEC;
  printf("Final routing length : %d\n", finallength);
  printf("Final number of via  : %d\n", numVia);
  printf("Final total length 1 : %d\n\n", finallength + numVia);

  // printf("Final total length 3 : %d\n",(finallength+3*numVia));
  // printf("3D runtime: %f sec\n", maze_Time);

  // freeAllMemory();
  return 0;
}
