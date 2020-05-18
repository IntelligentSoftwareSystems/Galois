/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <assert.h>
#include "cuda_launch_config.hpp"
#include "bh_tuning.h"

#define WARPSIZE 32
#define MAXDEPTH 32

__device__ volatile int stepd, bottomd, maxdepthd;
__device__ unsigned int blkcntd;
__device__ volatile float radiusd;


/******************************************************************************/
/*** initialize memory ********************************************************/
/******************************************************************************/

__global__ void InitializationKernel(int * __restrict errd)
{
  *errd = 0;
  stepd = -1;
  maxdepthd = 1;
  blkcntd = 0;
}


/******************************************************************************/
/*** compute center and radius ************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS1, FACTOR1)
void BoundingBoxKernel(int nnodesd, int nbodiesd, volatile int * __restrict startd, volatile int * __restrict childd, volatile float * __restrict massd, volatile float * __restrict posxd, volatile float * __restrict posyd, volatile float * __restrict poszd, volatile float * __restrict maxxd, volatile float * __restrict maxyd, volatile float * __restrict maxzd, volatile float * __restrict minxd, volatile float * __restrict minyd, volatile float * __restrict minzd)
{
  register int i, j, k, inc;
  register float val, minx, maxx, miny, maxy, minz, maxz;
  __shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1], sminz[THREADS1], smaxz[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];
  minz = maxz = poszd[0];

  // scan all bodies
  i = threadIdx.x;
  inc = THREADS1 * gridDim.x;
  for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc) {
    val = posxd[j];
    minx = fminf(minx, val);
    maxx = fmaxf(maxx, val);
    val = posyd[j];
    miny = fminf(miny, val);
    maxy = fmaxf(maxy, val);
    val = poszd[j];
    minz = fminf(minz, val);
    maxz = fmaxf(maxz, val);
  }

  // reduction in shared memory
  sminx[i] = minx;
  smaxx[i] = maxx;
  sminy[i] = miny;
  smaxy[i] = maxy;
  sminz[i] = minz;
  smaxz[i] = maxz;

  for (j = THREADS1 / 2; j > 0; j /= 2) {
    __syncthreads();
    if (i < j) {
      k = i + j;
      sminx[i] = minx = fminf(minx, sminx[k]);
      smaxx[i] = maxx = fmaxf(maxx, smaxx[k]);
      sminy[i] = miny = fminf(miny, sminy[k]);
      smaxy[i] = maxy = fmaxf(maxy, smaxy[k]);
      sminz[i] = minz = fminf(minz, sminz[k]);
      smaxz[i] = maxz = fmaxf(maxz, smaxz[k]);
    }
  }

  // write block result to global memory
  if (i == 0) {
    k = blockIdx.x;
    minxd[k] = minx;
    maxxd[k] = maxx;
    minyd[k] = miny;
    maxyd[k] = maxy;
    minzd[k] = minz;
    maxzd[k] = maxz;
    __threadfence();

    inc = gridDim.x - 1;
    if (inc == atomicInc(&blkcntd, inc)) {
      // I'm the last block, so combine all block results
      for (j = 0; j <= inc; j++) {
        minx = fminf(minx, minxd[j]);
        maxx = fmaxf(maxx, maxxd[j]);
        miny = fminf(miny, minyd[j]);
        maxy = fmaxf(maxy, maxyd[j]);
        minz = fminf(minz, minzd[j]);
        maxz = fmaxf(maxz, maxzd[j]);
      }

      // compute 'radius'
      val = fmaxf(maxx - minx, maxy - miny);
      radiusd = fmaxf(val, maxz - minz) * 0.5f;

      // create root node
      k = nnodesd;
      bottomd = k;

      massd[k] = -1.0f;
      startd[k] = 0;
      posxd[k] = (minx + maxx) * 0.5f;
      posyd[k] = (miny + maxy) * 0.5f;
      poszd[k] = (minz + maxz) * 0.5f;
      k *= 8;
      for (i = 0; i < 8; i++) childd[k + i] = -1;

      stepd++;
    }
  }
}


/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(1024, 1)
void ClearKernel1(int nnodesd, int nbodiesd, volatile int * __restrict childd)
{
  register int k, inc, top, bottom;

  top = 8 * nnodesd;
  bottom = 8 * nbodiesd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < top) {
    childd[k] = -1;
    k += inc;
  }
}


__global__
__launch_bounds__(THREADS2, FACTOR2)
void TreeBuildingKernel(int nnodesd, int nbodiesd, volatile int * __restrict errd, volatile int * __restrict childd, volatile float * __restrict posxd, volatile float * __restrict posyd, volatile float * __restrict poszd)
{
  register int i, j, depth, localmaxdepth, skip, inc;
  register float x, y, z, r;
  register float px, py, pz;
  register float dx, dy, dz;
  register int ch, n, cell, locked, patch;
  register float radius, rootx, rooty, rootz;

  // cache root data
  radius = radiusd;
  rootx = posxd[nnodesd];
  rooty = posyd[nnodesd];
  rootz = poszd[nnodesd];

  localmaxdepth = 1;
  skip = 1;
  inc = blockDim.x * gridDim.x;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < nbodiesd) {
    if (skip != 0) {
      // new body, so start traversing at root
      skip = 0;
      px = posxd[i];
      py = posyd[i];
      pz = poszd[i];
      n = nnodesd;
      depth = 1;
      r = radius * 0.5f;
      dx = dy = dz = -r;
      j = 0;
      // determine which child to follow
      if (rootx < px) {j = 1; dx = r;}
      if (rooty < py) {j |= 2; dy = r;}
      if (rootz < pz) {j |= 4; dz = r;}
      x = rootx + dx;
      y = rooty + dy;
      z = rootz + dz;
    }

    // follow path to leaf cell
    ch = childd[n*8+j];
    while (ch >= nbodiesd) {
      n = ch;
      depth++;
      r *= 0.5f;
      dx = dy = dz = -r;
      j = 0;
      // determine which child to follow
      if (x < px) {j = 1; dx = r;}
      if (y < py) {j |= 2; dy = r;}
      if (z < pz) {j |= 4; dz = r;}
      x += dx;
      y += dy;
      z += dz;
      ch = childd[n*8+j];
    }

    if (ch != -2) {  // skip if child pointer is locked and try again later
      locked = n*8+j;
      if (ch == -1) {
        if (-1 == atomicCAS((int *)&childd[locked], -1, i)) {  // if null, just insert the new body
          localmaxdepth = max(depth, localmaxdepth);
          i += inc;  // move on to next body
          skip = 1;
        }
      } else {  // there already is a body in this position
        if (ch == atomicCAS((int *)&childd[locked], ch, -2)) {  // try to lock
          patch = -1;
          // create new cell(s) and insert the old and new body
          do {
            depth++;

            cell = atomicSub((int *)&bottomd, 1) - 1;
            if (cell <= nbodiesd) {
              *errd = 1;
              bottomd = nnodesd;
            }

            if (patch != -1) {
              childd[n*8+j] = cell;
            }
            patch = max(patch, cell);

            j = 0;
            if (x < posxd[ch]) j = 1;
            if (y < posyd[ch]) j |= 2;
            if (z < poszd[ch]) j |= 4;
            childd[cell*8+j] = ch;

            n = cell;
            r *= 0.5f;
            dx = dy = dz = -r;
            j = 0;
            if (x < px) {j = 1; dx = r;}
            if (y < py) {j |= 2; dy = r;}
            if (z < pz) {j |= 4; dz = r;}
            x += dx;
            y += dy;
            z += dz;

            ch = childd[n*8+j];
            // repeat until the two bodies are different children
          } while (ch >= 0);
          childd[n*8+j] = i;

          localmaxdepth = max(depth, localmaxdepth);
          i += inc;  // move on to next body
          skip = 2;
        }
      }
    }
    __syncthreads();  // __threadfence();

    if (skip == 2) {
      childd[locked] = patch;
    }
  }
  // record maximum tree depth
  atomicMax((int *)&maxdepthd, localmaxdepth);
}


__global__
__launch_bounds__(1024, 1)
void ClearKernel2(int nnodesd, volatile int * __restrict startd, volatile float * __restrict massd)
{
  register int k, inc, bottom;

  bottom = bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < nnodesd) {
    massd[k] = -1.0f;
    startd[k] = -1;
    k += inc;
  }
}


/******************************************************************************/
/*** compute center of mass ***************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernel(const int nnodesd, const int nbodiesd, volatile int * __restrict countd, const int * __restrict childd, volatile float * __restrict massd, volatile float * __restrict posxd, volatile float * __restrict posyd, volatile float * __restrict poszd)
{
  register int i, j, k, ch, inc, cnt, bottom, flag;
  register float m, cm, px, py, pz;
  __shared__ int child[THREADS3 * 8];
  __shared__ float mass[THREADS3 * 8];

  bottom = bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  register int restart = k;
  for (j = 0; j < 5; j++) {  // wait-free pre-passes
    // iterate over all cells assigned to thread
    while (k <= nnodesd) {
      if (massd[k] < 0.0f) {
        for (i = 0; i < 8; i++) {
          ch = childd[k*8+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch >= nbodiesd) && ((mass[i*THREADS3+threadIdx.x] = massd[ch]) < 0.0f)) {
            break;
          }
        }
        if (i == 8) {
          // all children are ready
          cm = 0.0f;
          px = 0.0f;
          py = 0.0f;
          pz = 0.0f;
          cnt = 0;
          for (i = 0; i < 8; i++) {
            ch = child[i*THREADS3+threadIdx.x];
            if (ch >= 0) {
              if (ch >= nbodiesd) {  // count bodies (needed later)
                m = mass[i*THREADS3+threadIdx.x];
                cnt += countd[ch];
              } else {
                m = massd[ch];
                cnt++;
              }
              // add child's contribution
              cm += m;
              px += posxd[ch] * m;
              py += posyd[ch] * m;
              pz += poszd[ch] * m;
            }
          }
          countd[k] = cnt;
          m = 1.0f / cm;
          posxd[k] = px * m;
          posyd[k] = py * m;
          poszd[k] = pz * m;
          __threadfence();  // make sure data are visible before setting mass
          massd[k] = cm;
        }
      }
      k += inc;  // move on to next cell
    }
    k = restart;
  }

  flag = 0;
  j = 0;
  // iterate over all cells assigned to thread
  while (k <= nnodesd) {
    if (massd[k] >= 0.0f) {
      k += inc;
    } else {
      if (j == 0) {
        j = 8;
        for (i = 0; i < 8; i++) {
          ch = childd[k*8+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch < nbodiesd) || ((mass[i*THREADS3+threadIdx.x] = massd[ch]) >= 0.0f)) {
            j--;
          }
        }
      } else {
        j = 8;
        for (i = 0; i < 8; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if ((ch < nbodiesd) || (mass[i*THREADS3+threadIdx.x] >= 0.0f) || ((mass[i*THREADS3+threadIdx.x] = massd[ch]) >= 0.0f)) {
            j--;
          }
        }
      }

      if (j == 0) {
        // all children are ready
        cm = 0.0f;
        px = 0.0f;
        py = 0.0f;
        pz = 0.0f;
        cnt = 0;
        for (i = 0; i < 8; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if (ch >= 0) {
            if (ch >= nbodiesd) {  // count bodies (needed later)
              m = mass[i*THREADS3+threadIdx.x];
              cnt += countd[ch];
            } else {
              m = massd[ch];
              cnt++;
            }
            // add child's contribution
            cm += m;
            px += posxd[ch] * m;
            py += posyd[ch] * m;
            pz += poszd[ch] * m;
          }
        }
        countd[k] = cnt;
        m = 1.0f / cm;
        posxd[k] = px * m;
        posyd[k] = py * m;
        poszd[k] = pz * m;
        flag = 1;
      }
    }
    __syncthreads();  // __threadfence();
    if (flag != 0) {
      massd[k] = cm;
      k += inc;
      flag = 0;
    }
  }
}


/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS4, FACTOR4)
void SortKernel(int nnodesd, int nbodiesd, int * __restrict sortd, int * __restrict countd, volatile int * __restrict startd, int * __restrict childd)
{
  register int i, j, k, ch, dec, start, bottom;

  bottom = bottomd;
  dec = blockDim.x * gridDim.x;
  k = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all cells assigned to thread
  while (k >= bottom) {
    start = startd[k];
    if (start >= 0) {
      j = 0;
      for (i = 0; i < 8; i++) {
        ch = childd[k*8+i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front (needed later for speed)
            childd[k*8+i] = -1;
            childd[k*8+j] = ch;
          }
          j++;
          if (ch >= nbodiesd) {
            // child is a cell
            startd[ch] = start;  // set start ID of child
            start += countd[ch];  // add #bodies in subtree
          } else {
            // child is a body
            sortd[start] = ch;  // record body in 'sorted' array
            start++;
          }
        }
      }
      k -= dec;  // move on to next cell
    }
  }
}


/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernel(int nnodesd, int nbodiesd, volatile int * __restrict errd, float dthfd, float itolsqd, float epssqd, volatile int * __restrict sortd, volatile int * __restrict childd, volatile float * __restrict massd, volatile float * __restrict posxd, volatile float * __restrict posyd, volatile float * __restrict poszd, volatile float * __restrict velxd, volatile float * __restrict velyd, volatile float * __restrict velzd, volatile float * __restrict accxd, volatile float * __restrict accyd, volatile float * __restrict acczd)
{
  register int i, j, k, n, depth, base, sbase, diff, pd, nd;
  register float px, py, pz, ax, ay, az, dx, dy, dz, tmp;
  __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
  __shared__ float dq[MAXDEPTH * THREADS5/WARPSIZE];

  if (0 == threadIdx.x) {
    tmp = radiusd * 2;
    // precompute values that depend only on tree level
    dq[0] = tmp * tmp * itolsqd;
    for (i = 1; i < maxdepthd; i++) {
      dq[i] = dq[i - 1] * 0.25f;
      dq[i - 1] += epssqd;
    }
    dq[i - 1] += epssqd;

    if (maxdepthd > MAXDEPTH) {
      *errd = maxdepthd;
    }
  }
  __syncthreads();

  if (maxdepthd <= MAXDEPTH) {
    // figure out first thread in each warp (lane 0)
    base = threadIdx.x / WARPSIZE;
    sbase = base * WARPSIZE;
    j = base * MAXDEPTH;

    diff = threadIdx.x - sbase;
    // make multiple copies to avoid index calculations later
    if (diff < MAXDEPTH) {
      dq[diff+j] = dq[diff];
    }
    __syncthreads();
    __threadfence_block();

    // iterate over all bodies assigned to thread
    for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd; k += blockDim.x * gridDim.x) {
      i = sortd[k];  // get permuted/sorted index
      // cache position info
      px = posxd[i];
      py = posyd[i];
      pz = poszd[i];

      ax = 0.0f;
      ay = 0.0f;
      az = 0.0f;

      // initialize iteration stack, i.e., push root node onto stack
      depth = j;
      if (sbase == threadIdx.x) {
        pos[j] = 0;
        node[j] = nnodesd * 8;
      }

      do {
        // stack is not empty
        pd = pos[depth];
        nd = node[depth];
        while (pd < 8) {
          // node on top of stack has more children to process
          n = childd[nd + pd];  // load child pointer
          pd++;

          if (n >= 0) {
            dx = posxd[n] - px;
            dy = posyd[n] - py;
            dz = poszd[n] - pz;
            tmp = dx*dx + (dy*dy + (dz*dz + epssqd));  // compute distance squared (plus softening)
            if ((n < nbodiesd) || __all(tmp >= dq[depth])) {  // check if all threads agree that cell is far enough away (or is a body)
              tmp = rsqrtf(tmp);  // compute distance
              tmp = massd[n] * tmp * tmp * tmp;
              ax += dx * tmp;
              ay += dy * tmp;
              az += dz * tmp;
            } else {
              // push cell onto stack
              if (sbase == threadIdx.x) {  // maybe don't push and inc if last child
                pos[depth] = pd;
                node[depth] = nd;
              }
              depth++;
              pd = 0;
              nd = n * 8;
            }
          } else {
            pd = 8;  // early out because all remaining children are also zero
          }
        }
        depth--;  // done with this level
      } while (depth >= j);

      if (stepd > 0) {
        // update velocity
        velxd[i] += (ax - accxd[i]) * dthfd;
        velyd[i] += (ay - accyd[i]) * dthfd;
        velzd[i] += (az - acczd[i]) * dthfd;
      }

      // save computed acceleration
      accxd[i] = ax;
      accyd[i] = ay;
      acczd[i] = az;
    }
  }
}


/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS6, FACTOR6)
void IntegrationKernel(int nbodiesd, float dtimed, float dthfd, volatile float * __restrict posxd, volatile float * __restrict posyd, volatile float * __restrict poszd, volatile float * __restrict velxd, volatile float * __restrict velyd, volatile float * __restrict velzd, volatile float * __restrict accxd, volatile float * __restrict accyd, volatile float * __restrict acczd)
{
  register int i, inc;
  register float dvelx, dvely, dvelz;
  register float velhx, velhy, velhz;

  // iterate over all bodies assigned to thread
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += inc) {
    // integrate
    dvelx = accxd[i] * dthfd;
    dvely = accyd[i] * dthfd;
    dvelz = acczd[i] * dthfd;

    velhx = velxd[i] + dvelx;
    velhy = velyd[i] + dvely;
    velhz = velzd[i] + dvelz;

    posxd[i] += velhx * dtimed;
    posyd[i] += velhy * dtimed;
    poszd[i] += velhz * dtimed;

    velxd[i] = velhx + dvelx;
    velyd[i] = velhy + dvely;
    velzd[i] = velhz + dvelz;
  }
}


/******************************************************************************/

static void CudaTest(char *msg)
{
  cudaError_t e;

  cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}


/******************************************************************************/

// random number generator

#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;


static void drndset(int seed)
{
   A = 1;
   B = 0;
   randx = (A * seed + B) & MASK;
   A = (MULT * A) & MASK;
   B = (MULT * B + ADD) & MASK;
}


static double drnd()
{
   lastrand = randx;
   randx = (A * randx + B) & MASK;
   return (double)lastrand / TWOTO31;
}


/******************************************************************************/

int main(int argc, char *argv[])
{
  register int i, run, blocks;
  int nnodes, nbodies, step, timesteps;
  register double runtime;
  int error;
  register float dtime, dthf, epssq, itolsq;
  float time, timing[7];
  cudaEvent_t start, stop;
  float *mass, *posx, *posy, *posz, *velx, *vely, *velz;

  int *errl, *sortl, *childl, *countl, *startl;
  float *massl;
  float *posxl, *posyl, *poszl;
  float *velxl, *velyl, *velzl;
  float *accxl, *accyl, *acczl;
  float *maxxl, *maxyl, *maxzl;
  float *minxl, *minyl, *minzl;
  register double rsc, vsc, r, v, x, y, z, sq, scale;

  // perform some checks

  printf("CUDA BarnesHut v3.1 ");
#ifdef __KEPLER__
  printf("[Kepler]\n");
#else
  printf("[Fermi]\n");
#endif
  printf("Copyright (c) 2013, Texas State University-San Marcos. All rights reserved.\n");
  fflush(stdout);
  if (argc != 4) {
    fprintf(stderr, "\n");
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps device\n");
    exit(-1);
  }

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "There is no device supporting CUDA\n");
    exit(-1);
  }

  const int dev = atoi(argv[3]);
  if ((dev < 0) || (deviceCount <= dev)) {
    fprintf(stderr, "There is no device %d\n", dev);
    exit(-1);
  }
  cudaSetDevice(dev);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "There is no CUDA capable device\n");
    exit(-1);
  }
  if (deviceProp.major < 2) {
    fprintf(stderr, "Need at least compute capability 2.0\n");
    exit(-1);
  }
  if (deviceProp.warpSize != WARPSIZE) {
    fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
    exit(-1);
  }

  blocks = deviceProp.multiProcessorCount;
//  fprintf(stderr, "blocks = %d\n", blocks);

  if ((WARPSIZE <= 0) || (WARPSIZE & (WARPSIZE-1) != 0)) {
    fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
    exit(-1);
  }
  if (MAXDEPTH > WARPSIZE) {
    fprintf(stderr, "MAXDEPTH must be less than or equal to WARPSIZE\n");
    exit(-1);
  }
  if ((THREADS1 <= 0) || (THREADS1 & (THREADS1-1) != 0)) {
    fprintf(stderr, "THREADS1 must be greater than zero and a power of two\n");
    exit(-1);
  }

  // set L1/shared memory configuration
  cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
#ifdef __KEPLER__
  cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferEqual);
#else
  cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
#endif
  cudaFuncSetCacheConfig(IntegrationKernel, cudaFuncCachePreferL1);

  cudaGetLastError();  // reset error value
  for (run = 0; run < 3; run++) {
    for (i = 0; i < 7; i++) timing[i] = 0.0f;

    nbodies = atoi(argv[1]);
    if (nbodies < 1) {
      fprintf(stderr, "nbodies is too small: %d\n", nbodies);
      exit(-1);
    }
    if (nbodies > (1 << 30)) {
      fprintf(stderr, "nbodies is too large: %d\n", nbodies);
      exit(-1);
    }
    nnodes = nbodies * 2;
    if (nnodes < 1024*blocks) nnodes = 1024*blocks;
    while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
    nnodes--;

    timesteps = atoi(argv[2]);
    dtime = 0.025;  dthf = dtime * 0.5f;
    epssq = 0.05 * 0.05;
    itolsq = 1.0f / (0.5 * 0.5);

    // allocate memory

    if (run == 0) {
      printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

      mass = (float *)malloc(sizeof(float) * nbodies);
      if (mass == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
      posx = (float *)malloc(sizeof(float) * nbodies);
      if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
      posy = (float *)malloc(sizeof(float) * nbodies);
      if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
      posz = (float *)malloc(sizeof(float) * nbodies);
      if (posz == NULL) {fprintf(stderr, "cannot allocate posz\n");  exit(-1);}
      velx = (float *)malloc(sizeof(float) * nbodies);
      if (velx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
      vely = (float *)malloc(sizeof(float) * nbodies);
      if (vely == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}
      velz = (float *)malloc(sizeof(float) * nbodies);
      if (velz == NULL) {fprintf(stderr, "cannot allocate velz\n");  exit(-1);}

      if (cudaSuccess != cudaMalloc((void **)&errl, sizeof(int))) fprintf(stderr, "could not allocate errd\n");  CudaTest("couldn't allocate errd");
      if (cudaSuccess != cudaMalloc((void **)&childl, sizeof(int) * (nnodes+1) * 8)) fprintf(stderr, "could not allocate childd\n");  CudaTest("couldn't allocate childd");
      if (cudaSuccess != cudaMalloc((void **)&massl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate massd\n");  CudaTest("couldn't allocate massd");
      if (cudaSuccess != cudaMalloc((void **)&posxl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate posxd\n");  CudaTest("couldn't allocate posxd");
      if (cudaSuccess != cudaMalloc((void **)&posyl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate posyd\n");  CudaTest("couldn't allocate posyd");
      if (cudaSuccess != cudaMalloc((void **)&poszl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate poszd\n");  CudaTest("couldn't allocate poszd");
      if (cudaSuccess != cudaMalloc((void **)&velxl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate velxd\n");  CudaTest("couldn't allocate velxd");
      if (cudaSuccess != cudaMalloc((void **)&velyl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate velyd\n");  CudaTest("couldn't allocate velyd");
      if (cudaSuccess != cudaMalloc((void **)&velzl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate velzd\n");  CudaTest("couldn't allocate velzd");
      if (cudaSuccess != cudaMalloc((void **)&accxl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate accxd\n");  CudaTest("couldn't allocate accxd");
      if (cudaSuccess != cudaMalloc((void **)&accyl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate accyd\n");  CudaTest("couldn't allocate accyd");
      if (cudaSuccess != cudaMalloc((void **)&acczl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate acczd\n");  CudaTest("couldn't allocate acczd");
      if (cudaSuccess != cudaMalloc((void **)&countl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate countd\n");  CudaTest("couldn't allocate countd");
      if (cudaSuccess != cudaMalloc((void **)&startl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate startd\n");  CudaTest("couldn't allocate startd");
      if (cudaSuccess != cudaMalloc((void **)&sortl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate sortd\n");  CudaTest("couldn't allocate sortd");

      if (cudaSuccess != cudaMalloc((void **)&maxxl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate maxxd\n");  CudaTest("couldn't allocate maxxd");
      if (cudaSuccess != cudaMalloc((void **)&maxyl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate maxyd\n");  CudaTest("couldn't allocate maxyd");
      if (cudaSuccess != cudaMalloc((void **)&maxzl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate maxzd\n");  CudaTest("couldn't allocate maxzd");
      if (cudaSuccess != cudaMalloc((void **)&minxl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate minxd\n");  CudaTest("couldn't allocate minxd");
      if (cudaSuccess != cudaMalloc((void **)&minyl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate minyd\n");  CudaTest("couldn't allocate minyd");
      if (cudaSuccess != cudaMalloc((void **)&minzl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate minzd\n");  CudaTest("couldn't allocate minzd");
    }

    // generate input

    drndset(7);
    rsc = (3 * 3.1415926535897932384626433832795) / 16;
    vsc = sqrt(1.0 / rsc);
    for (i = 0; i < nbodies; i++) {
      mass[i] = 1.0 / nbodies;
      r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
      do {
        x = drnd()*2.0 - 1.0;
        y = drnd()*2.0 - 1.0;
        z = drnd()*2.0 - 1.0;
        sq = x*x + y*y + z*z;
      } while (sq > 1.0);
      scale = rsc * r / sqrt(sq);
      posx[i] = x * scale;
      posy[i] = y * scale;
      posz[i] = z * scale;

      do {
        x = drnd();
        y = drnd() * 0.1;
      } while (y > x*x * pow(1 - x*x, 3.5));
      v = x * sqrt(2.0 / sqrt(1 + r*r));
      do {
        x = drnd()*2.0 - 1.0;
        y = drnd()*2.0 - 1.0;
        z = drnd()*2.0 - 1.0;
        sq = x*x + y*y + z*z;
      } while (sq > 1.0);
      scale = vsc * v / sqrt(sq);
      velx[i] = x * scale;
      vely[i] = y * scale;
      velz[i] = z * scale;
    }

    if (cudaSuccess != cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of mass to device failed\n");  CudaTest("mass copy to device failed");
    if (cudaSuccess != cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posx to device failed\n");  CudaTest("posx copy to device failed");
    if (cudaSuccess != cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posy to device failed\n");  CudaTest("posy copy to device failed");
    if (cudaSuccess != cudaMemcpy(poszl, posz, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posz to device failed\n");  CudaTest("posz copy to device failed");
    if (cudaSuccess != cudaMemcpy(velxl, velx, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of velx to device failed\n");  CudaTest("velx copy to device failed");
    if (cudaSuccess != cudaMemcpy(velyl, vely, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of vely to device failed\n");  CudaTest("vely copy to device failed");
    if (cudaSuccess != cudaMemcpy(velzl, velz, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of velz to device failed\n");  CudaTest("velz copy to device failed");

    // run timesteps (launch GPU kernels)

    cudaEventCreate(&start);  cudaEventCreate(&stop);  
    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);

    cudaEventRecord(start, 0);
    InitializationKernel<<<1, 1>>>(errl);
    cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
    timing[0] += time;
    CudaTest("kernel 0 launch failed");

    for (step = 0; step < timesteps; step++) {
      //fprintf(stderr, "BBKernel\n");
      cudaEventRecord(start, 0);
      BoundingBoxKernel<<<blocks * FACTOR1, THREADS1>>>(nnodes, nbodies, startl, childl, massl, posxl, posyl, poszl, maxxl, maxyl, maxzl, minxl, minyl, minzl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[1] += time;
      CudaTest("kernel 1 launch failed");

      //fprintf(stderr, "TBKernel\n");
      cudaEventRecord(start, 0);
      ClearKernel1<<<blocks * 1, 1024>>>(nnodes, nbodies, childl);
      TreeBuildingKernel<<<blocks * FACTOR2, THREADS2>>>(nnodes, nbodies, errl, childl, posxl, posyl, poszl);
      ClearKernel2<<<blocks * 1, 1024>>>(nnodes, startl, massl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[2] += time;
      CudaTest("kernel 2 launch failed");

      //fprintf(stderr, "SKKernel\n");
      cudaEventRecord(start, 0);
      assert(all_resident(SummarizationKernel, dim3(blocks * FACTOR3), dim3(THREADS3), 0));
      SummarizationKernel<<<blocks * FACTOR3, THREADS3>>>(nnodes, nbodies, countl, childl, massl, posxl, posyl, poszl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[3] += time;
      CudaTest("kernel 3 launch failed");

      cudaEventRecord(start, 0);
      assert(all_resident(SortKernel, dim3(blocks * FACTOR4), dim3(THREADS4), 0));
      SortKernel<<<blocks * FACTOR4, THREADS4>>>(nnodes, nbodies, sortl, countl, startl, childl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[4] += time;
      CudaTest("kernel 4 launch failed");

      //fprintf(stderr, "FCKernel\n");
      cudaEventRecord(start, 0);
      ForceCalculationKernel<<<blocks * FACTOR5, THREADS5>>>(nnodes, nbodies, errl, dthf, itolsq, epssq, sortl, childl, massl, posxl, posyl, poszl, velxl, velyl, velzl, accxl, accyl, acczl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[5] += time;
      CudaTest("kernel 5 launch failed");

      //fprintf(stderr, "IKKernel\n");
      cudaEventRecord(start, 0);
      IntegrationKernel<<<blocks * FACTOR6, THREADS6>>>(nbodies, dtime, dthf, posxl, posyl, poszl, velxl, velyl, velzl, accxl, accyl, acczl);
      cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
      timing[6] += time;
      CudaTest("kernel 6 launch failed");
    }
    CudaTest("kernel launch failed");
    cudaEventDestroy(start);  cudaEventDestroy(stop);

    // transfer result back to CPU
    if (cudaSuccess != cudaMemcpy(&error, errl, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of err from device failed\n");  CudaTest("err copy from device failed");
    if (cudaSuccess != cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posx from device failed\n");  CudaTest("posx copy from device failed");
    if (cudaSuccess != cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posy from device failed\n");  CudaTest("posy copy from device failed");
    if (cudaSuccess != cudaMemcpy(posz, poszl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posz from device failed\n");  CudaTest("posz copy from device failed");
    if (cudaSuccess != cudaMemcpy(velx, velxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of velx from device failed\n");  CudaTest("velx copy from device failed");
    if (cudaSuccess != cudaMemcpy(vely, velyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of vely from device failed\n");  CudaTest("vely copy from device failed");
    if (cudaSuccess != cudaMemcpy(velz, velzl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of velz from device failed\n");  CudaTest("velz copy from device failed");

    gettimeofday(&endtime, NULL);
    runtime = endtime.tv_sec + endtime.tv_usec/1000000.0 - starttime.tv_sec - starttime.tv_usec/1000000.0;

    printf("runtime: %.4lf s  (", runtime);
    time = 0;
    for (i = 1; i < 7; i++) {
      printf(" %.1f ", timing[i]);
      time += timing[i];
    }
    if (error == 0) {
      printf(") = %.1f ms\n", time);
    } else {
      printf(") = %.1f ms FAILED %d\n", time, error);
    }
  }

  // print output
  i = 0;
//  for (i = 0; i < nbodies; i++) {
    printf("%.2e %.2e %.2e\n", posx[i], posy[i], posz[i]);
//  }

  free(mass);
  free(posx);
  free(posy);
  free(posz);
  free(velx);
  free(vely);
  free(velz);

  cudaFree(errl);
  cudaFree(childl);
  cudaFree(massl);
  cudaFree(posxl);
  cudaFree(posyl);
  cudaFree(poszl);
  cudaFree(countl);
  cudaFree(startl);

  cudaFree(maxxl);
  cudaFree(maxyl);
  cudaFree(maxzl);
  cudaFree(minxl);
  cudaFree(minyl);
  cudaFree(minzl);

  return 0;
}
