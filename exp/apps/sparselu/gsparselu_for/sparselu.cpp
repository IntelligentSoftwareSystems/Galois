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
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

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
void genmat (float *M[])
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
	 if (ii==jj-1) null_entry = FALSE; // XXX comment this out to get non-bulk sync structure
         if (ii-1 == jj) null_entry = FALSE; // XXX ditto
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

void sparselu_init (float ***pBENCH, char *pass)
{
   Galois::setActiveThreads(bots_arg_size_2);
   Galois::Runtime::getSystemThreadPool().burnPower(bots_arg_size_2);
   Galois::preAlloc(5*bots_arg_size_2);
   Galois::reportPageAlloc("MeminfoPre");
   *pBENCH = (float **) malloc(bots_arg_size*bots_arg_size*sizeof(float *));
   genmat(*pBENCH);
   print_structure(pass, *pBENCH);
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

void sparselu_par_call(float **BENCH)
{
  int ii, jj, kk;

  Galois::StatManager manager;

  bots_message("Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ",
          bots_arg_size,bots_arg_size,bots_arg_size_1,bots_arg_size_1);

  namespace ww = Galois::WorkList;
  typedef Galois::WorkList::StableIterator<>::with_container<ww::dChunkedLIFO<1>>::type WL;

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
  bots_message(" completed!\n");
}

void sparselu_fini (float **BENCH, char *pass)
{
   Galois::Runtime::getSystemThreadPool().beKind();
   Galois::reportPageAlloc("MeminfoPost");
   print_structure(pass, BENCH);
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

