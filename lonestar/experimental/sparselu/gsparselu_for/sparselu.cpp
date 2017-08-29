/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <libgen.h>
#include "bots.h"
#include "sparselu.h"

#include "Galois/Galois.h"
#include "Galois/Statistic.h"
#include "Galois/LargeArray.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Runtime/KDGtwoPhase.h"
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

extern char bots_arg_file[256];

/***********************************************************************
 * checkmat: 
 **********************************************************************/
int checkmat (float *M, float *N)
{
   int i, j;
   float r_err;

   for (i = 0; i < bots_arg_size_1; i++) 
   {
      for (j = 0; j < bots_arg_size_1; j++) 
      {
         r_err = M[i*bots_arg_size_1+j] - N[i*bots_arg_size_1+j];
         if ( r_err == 0.0 ) continue;

         if (r_err < 0.0 ) r_err = -r_err;

         if ( M[i*bots_arg_size_1+j] == 0 )
         {
           bots_message("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; \n",
                    i,j, M[i*bots_arg_size_1+j], i,j, N[i*bots_arg_size_1+j]);
           return FALSE;
         }
         r_err = r_err / M[i*bots_arg_size_1+j];
         if(r_err > EPSILON)
         {
            bots_message("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; Relative Error=%f\n",
                    i,j, M[i*bots_arg_size_1+j], i,j, N[i*bots_arg_size_1+j], r_err);
            return FALSE;
         }
      }
   }
   return TRUE;
}

/***********************************************************************
 * genmat: 
 **********************************************************************/
static void synthetic_genmat (float *M[])
{
   int null_entry, init_val, i, j, ii, jj;
   float *p;
   int a=0,b=0;

   init_val = 1325;

   /* generating the structure */
   for (ii=0; ii < bots_arg_size; ii++)
   {
      for (jj=0; jj < bots_arg_size; jj++)
      {
         /* computing null entries */
         null_entry=FALSE;
         if ((ii<jj) && (ii%3 !=0)) null_entry = TRUE;
         if ((ii>jj) && (jj%3 !=0)) null_entry = TRUE;
	 if (ii%2==1) null_entry = TRUE;
	 if (jj%2==1) null_entry = TRUE;
	 if (ii==jj) null_entry = FALSE;
	 if (ii==jj-1) null_entry = FALSE;
         if (ii-1 == jj) null_entry = FALSE;
         /* allocating matrix */
         if (null_entry == FALSE){
            a++;
            M[ii*bots_arg_size+jj] = (float *) malloc(bots_arg_size_1*bots_arg_size_1*sizeof(float));
	    if (M[ii*bots_arg_size+jj] == NULL)
            {
               bots_message("Error: Out of memory\n");
               exit(101);
            }
            /* initializing matrix */
            p = M[ii*bots_arg_size+jj];
            for (i = 0; i < bots_arg_size_1; i++) 
            {
               for (j = 0; j < bots_arg_size_1; j++)
               {
	            init_val = (3125 * init_val) % 65536;
      	            (*p) = (float)((init_val - 32768.0) / 16384.0);
                    p++;
               }
            }
         }
         else
         {
            b++;
            M[ii*bots_arg_size+jj] = NULL;
         }
      }
   }
   bots_debug("allo = %d, no = %d, total = %d, factor = %f\n",a,b,a+b,(float)((float)a/(float)(a+b)));
}


// From symmetric matrix
static void structure_from_file_genmat (float *M[])
{
   int ii, jj;
   int a=0, b;
   Galois::Graph::FileGraph g;
   int num_blocks;
   unsigned max_id;

   g.fromFile(bots_arg_file);
   memset(M, 0, bots_arg_size*bots_arg_size*sizeof(*M));

   num_blocks = (g.size() + bots_arg_size_1 - 1) / bots_arg_size_1;
   max_id = bots_arg_size_1 * bots_arg_size;

   printf("full size: %d\n", num_blocks);

   /* generating the structure */
   for (auto ii : g)
   {
      if (ii >= max_id)
         break;
      int bii = ii / bots_arg_size_1;
      for (auto edge : g.out_edges(ii))
      {
         /* computing null entries */
         int jj = g.getEdgeDst(edge);
         if (jj >= max_id)
            continue;
         int bjj = jj / bots_arg_size_1;
         if (M[bii*bots_arg_size+bjj] == NULL)
         {
            a++;
            M[bii*bots_arg_size+bjj] = (float *) malloc(bots_arg_size_1*bots_arg_size_1*sizeof(float));
            memset(M[bii*bots_arg_size+bjj], 0, bots_arg_size_1*bots_arg_size_1*sizeof(float));
         }
         if (M[bii*bots_arg_size+bjj] == NULL)
         {
            bots_message("Error: Out of memory\n");
            exit(101);
         }
         if (M[bjj*bots_arg_size+bii] == NULL)
         {
            a++;
            M[bjj*bots_arg_size+bii] = (float *) malloc(bots_arg_size_1*bots_arg_size_1*sizeof(float));
            memset(M[bjj*bots_arg_size+bii], 0, bots_arg_size_1*bots_arg_size_1*sizeof(float));
         }
         if (M[bjj*bots_arg_size+bii] == NULL)
         {
            bots_message("Error: Out of memory\n");
            exit(101);
         }
         M[bii*bots_arg_size+bjj][(ii%bots_arg_size_1)*bots_arg_size_1+(jj%bots_arg_size_1)] = g.getEdgeData<float>(edge);
         M[bjj*bots_arg_size+bii][(jj%bots_arg_size_1)*bots_arg_size_1+(ii%bots_arg_size_1)] = g.getEdgeData<float>(edge);
      }
   }
   // Add identity diagonal as necessary
   for (ii = 0; ii < bots_arg_size; ++ii) {
     if (M[ii*bots_arg_size+ii] == NULL)
     {
        a++;
        M[ii*bots_arg_size+ii] = (float *) malloc(bots_arg_size_1*bots_arg_size_1*sizeof(float));
        memset(M[ii*bots_arg_size+ii], 0, bots_arg_size_1*bots_arg_size_1*sizeof(float));
     }
     for (jj = 0; jj < bots_arg_size_1; ++jj)
     {
       if (M[ii*bots_arg_size+ii][jj*bots_arg_size_1+jj] == 0.0)
         M[ii*bots_arg_size+ii][jj*bots_arg_size_1+jj] = 1.0;
     }
   }
   b = num_blocks * num_blocks - a;
   bots_debug("allo = %d, no = %d, total = %d, factor = %f\n",a,b,a+b,(float)((float)a/(float)(a+b)));
}

void genmat (float *M[])
{
  if (strlen(bots_arg_file) == 0)
    synthetic_genmat(M);
  else
    structure_from_file_genmat(M);
}


/***********************************************************************
 * print_structure: 
 **********************************************************************/
void print_structure(char *name, float *M[])
{
   int ii, jj;
   bots_message("Structure for matrix %s @ 0x%p\n",name, M);
   for (ii = 0; ii < bots_arg_size; ii++) {
     for (jj = 0; jj < bots_arg_size; jj++) {
        if (M[ii*bots_arg_size+jj]!=NULL) {bots_message("x");}
        else bots_message(" ");
     }
     bots_message("\n");
   }
   bots_message("\n");
}
/***********************************************************************
 * allocate_clean_block: 
 **********************************************************************/
float * allocate_clean_block()
{
  int i,j;
  float *p, *q;

  p = (float *) malloc(bots_arg_size_1*bots_arg_size_1*sizeof(float));
  q=p;
  if (p!=NULL){
     for (i = 0; i < bots_arg_size_1; i++) 
        for (j = 0; j < bots_arg_size_1; j++){(*p)=0.0; p++;}
	
  }
  else
  {
      bots_message("Error: Out of memory\n");
      exit (101);
  }
  return (q);
}

/***********************************************************************
 * lu0: 
 **********************************************************************/
void lu0(float *diag)
{
   int i, j, k;

   for (k=0; k<bots_arg_size_1; k++)
      for (i=k+1; i<bots_arg_size_1; i++)
      {
         diag[i*bots_arg_size_1+k] = diag[i*bots_arg_size_1+k] / diag[k*bots_arg_size_1+k];
         for (j=k+1; j<bots_arg_size_1; j++)
            diag[i*bots_arg_size_1+j] = diag[i*bots_arg_size_1+j] - diag[i*bots_arg_size_1+k] * diag[k*bots_arg_size_1+j];
      }
}

/***********************************************************************
 * bdiv: 
 **********************************************************************/
void bdiv(float *diag, float *row)
{
   int i, j, k;
   for (i=0; i<bots_arg_size_1; i++)
      for (k=0; k<bots_arg_size_1; k++)
      {
         row[i*bots_arg_size_1+k] = row[i*bots_arg_size_1+k] / diag[k*bots_arg_size_1+k];
         for (j=k+1; j<bots_arg_size_1; j++)
            row[i*bots_arg_size_1+j] = row[i*bots_arg_size_1+j] - row[i*bots_arg_size_1+k]*diag[k*bots_arg_size_1+j];
      }
}
/***********************************************************************
 * bmod: 
 **********************************************************************/
void bmod(float *row, float *col, float *inner)
{
   int i, j, k;
   for (i=0; i<bots_arg_size_1; i++)
      for (j=0; j<bots_arg_size_1; j++)
         for (k=0; k<bots_arg_size_1; k++)
            inner[i*bots_arg_size_1+j] = inner[i*bots_arg_size_1+j] - row[i*bots_arg_size_1+k]*col[k*bots_arg_size_1+j];
}
/***********************************************************************
 * fwd: 
 **********************************************************************/
void fwd(float *diag, float *col)
{
   int i, j, k;
   for (j=0; j<bots_arg_size_1; j++)
      for (k=0; k<bots_arg_size_1; k++) 
         for (i=k+1; i<bots_arg_size_1; i++)
            col[i*bots_arg_size_1+j] = col[i*bots_arg_size_1+j] - diag[i*bots_arg_size_1+k]*col[k*bots_arg_size_1+j];
}

static Galois::LargeArray<Galois::Runtime::Lockable> locks;

void sparselu_init (float ***pBENCH, char *pass)
{
   Galois::setActiveThreads(bots_arg_size_2);
   Galois::Substrate::getThreadPool().burnPower(bots_arg_size_2);
   Galois::preAlloc(5*bots_arg_size_2);
   Galois::reportPageAlloc("MeminfoPre");
   *pBENCH = (float **) malloc(bots_arg_size*bots_arg_size*sizeof(float *));
   genmat(*pBENCH);
   print_structure(pass, *pBENCH);
   //locks.allocateInterleaved(bots_arg_size*bots_arg_size); XXX
   locks.allocateLocal(bots_arg_size*bots_arg_size);
   locks.construct();
}

void sparselu_seq_call(float **BENCH)
{
   int ii, jj, kk;

   for (kk=0; kk<bots_arg_size; kk++)
   {
      lu0(BENCH[kk*bots_arg_size+kk]);
      for (jj=kk+1; jj<bots_arg_size; jj++)
         if (BENCH[kk*bots_arg_size+jj] != NULL)
         {
            fwd(BENCH[kk*bots_arg_size+kk], BENCH[kk*bots_arg_size+jj]);
         }
      for (ii=kk+1; ii<bots_arg_size; ii++) 
         if (BENCH[ii*bots_arg_size+kk] != NULL)
         {
            bdiv (BENCH[kk*bots_arg_size+kk], BENCH[ii*bots_arg_size+kk]);
         }
      for (ii=kk+1; ii<bots_arg_size; ii++)
         if (BENCH[ii*bots_arg_size+kk] != NULL)
            for (jj=kk+1; jj<bots_arg_size; jj++)
               if (BENCH[kk*bots_arg_size+jj] != NULL)
               {
                     if (BENCH[ii*bots_arg_size+jj]==NULL) BENCH[ii*bots_arg_size+jj] = allocate_clean_block();
                     bmod(BENCH[ii*bots_arg_size+kk], BENCH[kk*bots_arg_size+jj], BENCH[ii*bots_arg_size+jj]);
               }

   }
}

struct FwdBdiv {
  typedef int tt_does_not_need_aborts;

  float** BENCH;
  int kk;

  enum { SEEK, FWD, BDIV };

  struct Task {
    int type;
    int arg;
  };

  struct Initializer: public std::unary_function<int, Task> {
    Task operator()(int arg) const {
      return { SEEK, arg };
    }
  };

  void doSeek(const Task& t, Galois::UserContext<Task>& ctx) {
    int jj = t.arg;
    if (BENCH[kk*bots_arg_size+jj] != NULL)
      ctx.push(Task {FWD, jj});
    int ii = t.arg;
    if (BENCH[ii*bots_arg_size+kk] != NULL)
      ctx.push(Task {BDIV, ii});
  }

  void doFwd(const Task& t, Galois::UserContext<Task>& ctx) {
    int jj = t.arg;
    fwd(BENCH[kk*bots_arg_size+kk], BENCH[kk*bots_arg_size+jj]);
  }

  void doBdiv(const Task& t, Galois::UserContext<Task>& ctx) {
    int ii = t.arg;
    bdiv(BENCH[kk*bots_arg_size+kk], BENCH[ii*bots_arg_size+kk]);
  }

  void operator()(const Task& t, Galois::UserContext<Task>& ctx) {
    switch (t.type) {
      case SEEK: return doSeek(t, ctx);
      case FWD: return doFwd(t, ctx);
      case BDIV: return doBdiv(t, ctx);
      default: abort();
    }
  }
};

struct Bmod {
  typedef int tt_does_not_need_aborts;

  float** BENCH;
  int kk;

  enum { SEEK, BMOD };

  struct Task {
    int type;
    int arg1;
    int arg2;
  };

  struct Initializer: public std::unary_function<int, Task> {
    Task operator()(int arg) const {
      return { SEEK, arg, arg };
    }
  };

  void doSeek(const Task& t, Galois::UserContext<Task>& ctx) {
    int ii = t.arg1;
    if (BENCH[ii*bots_arg_size+kk] != NULL)
      for (int jj=kk+1; jj<bots_arg_size; jj++)
        if (BENCH[kk*bots_arg_size+jj] != NULL)
          ctx.push(Task { BMOD, ii, jj });
  }

  void doBmod(const Task& t, Galois::UserContext<Task>& ctx) {
    int ii = t.arg1;
    int jj = t.arg2;
    if (BENCH[ii*bots_arg_size+jj]==NULL)
      BENCH[ii*bots_arg_size+jj] = allocate_clean_block();
    bmod(BENCH[ii*bots_arg_size+kk], BENCH[kk*bots_arg_size+jj], BENCH[ii*bots_arg_size+jj]);
  }

  void operator()(const Task& t, Galois::UserContext<Task>& ctx) {
    switch (t.type) {
      case SEEK: return doSeek(t, ctx);
      case BMOD: return doBmod(t, ctx);
      default: abort();
    }
  }
};

struct FwdBdivBmod {
  enum { LU0=0, FWD=1, BDIV=2, BMOD=3 };

  struct Task {
    int type;
    int kk;
    int arg0;
    int arg1;
  };

  struct Initializer: public std::unary_function<int, Task> {
    Task operator()(int kk) const { return { LU0, kk, 0, 0 }; }
  };

  struct Comparator {
    bool operator()(const Task& t1, const Task& t2) const {
      // Lexicographic (kk, type, arg0, arg1)
      if (t1.kk == t2.kk)
        if (t1.type == t2.type)
          if (t1.arg0 == t2.arg0)
            return t1.arg1 < t2.arg1;
          else
            return t1.arg0 < t2.arg0;
        else
          return t1.type < t2.type;
      else
        return t1.kk < t2.kk;
    }
  };

  struct NeighborhoodVisitor {
    static const unsigned CHUNK_SIZE = 4;
    float** BENCH;

    typedef int tt_does_not_need_push;

    void acquire(int ii, int jj) {
      Galois::Runtime::acquire(&locks[ii*bots_arg_size+jj], Galois::MethodFlag::WRITE);
    }

    void operator()(const Task& t, Galois::UserContext<Task>&) {
      int ii, jj, kk = t.kk;
      switch (t.type) {
        case LU0:
          acquire(kk, kk);
          for (jj=kk+1; jj<bots_arg_size; jj++) {
             if (BENCH[kk*bots_arg_size+jj] != NULL) {
                acquire(kk, jj);
             }
          }
          for (ii=kk+1; ii<bots_arg_size; ii++) {
             if (BENCH[ii*bots_arg_size+kk] != NULL) {
               acquire(ii, kk);
             }
          }
          for (ii=kk+1; ii<bots_arg_size; ii++) {
             if (BENCH[ii*bots_arg_size+kk] != NULL) {
                acquire(ii, kk);
                for (jj=kk+1; jj<bots_arg_size; jj++) {
                   if (BENCH[kk*bots_arg_size+jj] != NULL) {
                     acquire(kk, jj);
                     acquire(ii, jj);
                   }
                }
             }
          }
          return;
        case FWD:
          acquire(kk, t.arg0);
          for (ii=kk+1; ii<bots_arg_size; ii++) {
             if (BENCH[ii*bots_arg_size+kk] != NULL) {
                acquire(ii, t.arg0);
             }
          }
          return;
        case BDIV:
          acquire(t.arg0, kk);
          for (jj=kk+1; jj<bots_arg_size; jj++) {
             if (BENCH[kk*bots_arg_size+jj] != NULL) {
                acquire(t.arg0, jj);
             }
          }
          return;
        case BMOD:
          acquire(t.arg0, t.arg1);
          return;
      }
    }
  };

  struct Process {
    static const int CHUNK_SIZE = 1;

    typedef int tt_does_not_need_aborts;

    float** BENCH;

    void operator()(const Task& t, Galois::UserContext<Task>& ctx) {
      int ii, jj, kk = t.kk;

      switch (t.type) {
        case LU0:
          lu0(BENCH[kk*bots_arg_size+kk]);
          for (ii=kk+1; ii<bots_arg_size; ii++) {
             if (BENCH[ii*bots_arg_size+kk] != NULL) {
                ctx.push(Task { BDIV, kk, ii, 0 });
             }
          }
          for (jj=kk+1; jj<bots_arg_size; jj++) {
             if (BENCH[kk*bots_arg_size+jj] != NULL) {
                ctx.push(Task { FWD, kk, jj, 0 });
             }
          }
          for (ii=kk+1; ii<bots_arg_size; ii++) {
             if (BENCH[ii*bots_arg_size+kk] != NULL) {
                for (jj=kk+1; jj<bots_arg_size; jj++) {
                   if (BENCH[kk*bots_arg_size+jj] != NULL) {
                      if (BENCH[ii*bots_arg_size+jj]==NULL)
                         BENCH[ii*bots_arg_size+jj] = allocate_clean_block();
                      ctx.push(Task { BMOD, kk, ii, jj });
                   }
                }
             }
          }
          return;
        case FWD:
          return fwd(BENCH[kk*bots_arg_size+kk], BENCH[kk*bots_arg_size+t.arg0]);
        case BDIV:
          return bdiv(BENCH[kk*bots_arg_size+kk], BENCH[t.arg0*bots_arg_size+kk]);
        case BMOD:
          return bmod(BENCH[t.arg0*bots_arg_size+kk], BENCH[kk*bots_arg_size+t.arg1], BENCH[t.arg0*bots_arg_size+t.arg1]);
      }
    }
  };
};

static void bs_algo(float **BENCH)
{
  int ii, jj, kk;

  namespace ww = Galois::WorkList;
  typedef ww::StableIterator<>::with_container<ww::dChunkedLIFO<1>>::type WL;

  for (kk=0; kk<bots_arg_size; kk++) {
    lu0(BENCH[kk*bots_arg_size+kk]);

    Galois::for_each(
        boost::transform_iterator<FwdBdiv::Initializer, boost::counting_iterator<int>>(kk+1),
        boost::transform_iterator<FwdBdiv::Initializer, boost::counting_iterator<int>>(bots_arg_size),
        FwdBdiv { BENCH, kk }, Galois::wl<WL>());

    Galois::for_each(
        boost::transform_iterator<Bmod::Initializer, boost::counting_iterator<int>>(kk+1),
        boost::transform_iterator<Bmod::Initializer, boost::counting_iterator<int>>(bots_arg_size),
        Bmod { BENCH, kk }, Galois::wl<WL>());
  }
}

static void ikdg_algo(float **BENCH)
{
  int kk;
  typedef boost::transform_iterator<FwdBdivBmod::Initializer, boost::counting_iterator<int>> TI;

  Galois::setDoAllImpl (Galois::DOALL_COUPLED);
  if (Galois::getDoAllImpl () != Galois::DOALL_COUPLED) { std::abort (); }

  Galois::Runtime::for_each_ordered_ikdg(
      Galois::Runtime::makeStandardRange(TI(0), TI(bots_arg_size)),
      FwdBdivBmod::Comparator { },
      FwdBdivBmod::NeighborhoodVisitor { BENCH },
      FwdBdivBmod::Process { BENCH },
      std::make_tuple(
        Galois::loopname("sparselu-ikdg")));
}

void sparselu_par_call(float **BENCH)
{
  Galois::StatManager manager;
  Galois::StatTimer T;

  T.start();
  bots_message("Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ",
          bots_arg_size,bots_arg_size,bots_arg_size_1,bots_arg_size_1);
  if (bots_app_cutoff_value_1 == 0)
    bs_algo(BENCH);
  else
    ikdg_algo(BENCH);
  bots_message(" completed!\n");
  Galois::reportPageAlloc("MeminfoPost");
  T.stop();
}

void sparselu_fini (float **BENCH, char *pass)
{
  Galois::Substrate::getThreadPool().beKind();
   print_structure(pass, BENCH);
   locks.destroy();
   locks.deallocate();
}

int sparselu_check(float **SEQ, float **BENCH)
{
   int ii,jj,ok=1;

   for (ii=0; ((ii<bots_arg_size) && ok); ii++)
   {
      for (jj=0; ((jj<bots_arg_size) && ok); jj++)
      {
         if (SEQ[ii*bots_arg_size+jj] == NULL && BENCH[ii*bots_arg_size+jj] != NULL) ok = FALSE;
         if (SEQ[ii*bots_arg_size+jj] != NULL && BENCH[ii*bots_arg_size+jj] == NULL) ok = FALSE;
         if (SEQ[ii*bots_arg_size+jj] != NULL && BENCH[ii*bots_arg_size+jj] != NULL)
            ok = checkmat(SEQ[ii*bots_arg_size+jj], BENCH[ii*bots_arg_size+jj]);
      }
   }
   if (ok) return BOTS_RESULT_SUCCESSFUL;
   else return BOTS_RESULT_UNSUCCESSFUL;
}

