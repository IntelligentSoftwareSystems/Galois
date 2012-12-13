#ifndef GALOIS_RUNTIME_H
#define GALOIS_RUNTIME_H

#include "mpi.h"
#include <stdlib.h>
#include <iostream>
#include "Galois/Galois.h"

/* buffer sizes */
#define RSIZ      (512*sizeof(int))

/* types of tags */
#define CMD_TAG   0x1
#define STOP_TAG  0x2
#define DATA_TAG  0x3
#define FUNC_TAG  0x4
#define COUNT_TAG 0x5
#define CLASS_TAG 0x6
#define END_TAG   0x7

using namespace std;

typedef void (*ftype) (char*,int,char*);

namespace Galois::Runtime {

   int rank, numtasks;

   void task_synchronize() {
      /* Call the MPI_Barrier function */
      if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) {
         cout << "Error in MPI_Barrier " << __FILE__ << __LINE__ << endl;
         MPI_Finalize();
         exit (1);
      }
      return;
   }

   void task_terminate() {
      int  numTasks = DIR::getNoTasks();
      int  taskRank = DIR::getTaskRank();
      int  tasksCompleted = 1;
      bool *complist;
      int  flag;
      MPI_Request sreq, rreq;
      MPI_Status  status;

      // send a completed message to every task
      complist = (bool*)malloc(numTasks*sizeof(bool));
      for (int i = 0; i < numTasks; i++) {
         complist[i] = false;
         if (i == taskRank)
           continue;
         MPI_Isend (&taskRank, 1, MPI_INT, i, END_TAG, MPI_COMM_WORLD, &sreq);
      }
      complist[taskRank] = true;
      MPI_Irecv (&taskRank, 1, MPI_INT, MPI_ANY_SOURCE, END_TAG, MPI_COMM_WORLD, &rreq);

      // check if any new completion message has been received
      do {
   //    usleep(1);
         DIR::comm();
         MPI_Test (&rreq, &flag, &status);
         if (flag) {
            if (!complist[taskRank]) {
               complist[taskRank] = true;
               tasksCompleted++;
            }
            MPI_Irecv (&taskRank, 1, MPI_INT, MPI_ANY_SOURCE, END_TAG, MPI_COMM_WORLD, &rreq);
         }
      } while(numTasks != tasksCompleted);

      free (complist);
      return;
   }

   /* This is the work loop from which tasks never return */
   void task_worker() {
      bool    flag = true;
      int     received;
      int     data_count = 0;
      char    buf[RSIZ], data[RSIZ], class_data[RSIZ], stbuf[RSIZ];
      size_t *sbuf = (size_t*)buf;
      ftype   addr;
      int     err;

      MPI_Status  status;
      MPI_Request creq, creq1;

      task_synchronize();

      /* receive the command from the master (rank = 0) */
      err = MPI_Irecv(buf, RSIZ, MPI_BYTE, 0, COUNT_TAG, MPI_COMM_WORLD, &creq);
      err = MPI_Irecv(stbuf, RSIZ, MPI_BYTE, 0, STOP_TAG, MPI_COMM_WORLD, &creq1);

      do {
         /* check if asked to stop */
         err = MPI_Test(&creq1, &received, &status);
         if (received) {
            /* stop message received */
            break;
         }
         err = MPI_Test(&creq, &received, &status);
         if (!received) {
            /* No message received */
            DIR::comm();
            continue;
         }
         switch (status.MPI_TAG) {
            case DATA_TAG:
               memcpy (data, buf, RSIZ);
               err = MPI_Irecv(buf, RSIZ, MPI_BYTE, 0, CLASS_TAG, MPI_COMM_WORLD, &creq);
               break;
            case FUNC_TAG:
               addr = (ftype)sbuf[0];
               addr(class_data,data_count,data);
               task_terminate();
               /* barrier to synchronize after command */
               task_synchronize();
               err = MPI_Irecv(buf, RSIZ, MPI_BYTE, 0, COUNT_TAG, MPI_COMM_WORLD, &creq);
               break;
            case COUNT_TAG:
               data_count = *((int*)buf);
               err = MPI_Irecv(buf, RSIZ, MPI_BYTE, 0, DATA_TAG, MPI_COMM_WORLD, &creq);
               break;
            case CLASS_TAG:
               memcpy (class_data, buf, RSIZ);
               err = MPI_Irecv(buf, RSIZ, MPI_BYTE, 0, FUNC_TAG, MPI_COMM_WORLD, &creq);
               break;
            default:
               cout << rank << " received unrecognized TAG! " << __FILE__;
               cout << ":" << __LINE__ << endl;
               break;
         }
      } while (flag);

      /* exit as the worker task should not return */
      MPI_Finalize();
      exit(0);

      return;
   }

   void for_each_begin_impl() {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

      /* if rank != 0 wait for work to get assigned */
      if (rank != 0)
         task_worker();
      else
         task_synchronize();

      return;
   }

   template<typename IterTy, typename T, typename FunctionTy>
   void master_send_data(IterTy b, IterTy e, FunctionTy f) {
      T       buf[RSIZ];
      char   *rbuf = (char*)buf;
      size_t *sbuf = (size_t*)buf;
      MPI_Request sreq;
      int count, task_count;

      /* count number of elements */
      count = std::distance(b,e);

      /* calculate the num of elements to be sent to each task */
      task_count = count/numtasks;
      for (int i = 1; i < numtasks; i++) {
         int *tmp = (int*)rbuf;
         *tmp = task_count;
         MPI_Isend (rbuf, RSIZ, MPI_BYTE, i, COUNT_TAG, MPI_COMM_WORLD, &sreq);
      }

      /* send the data */
      IterTy send_it = b;
      for (int i = 1; i < numtasks; i++) {
         /* pack the data */
         for (int j = 0; j < task_count; j++) {
            buf[j] = *send_it;
            send_it++;
         }
         /* send the data */
         MPI_Isend (rbuf, RSIZ, MPI_BYTE, i, DATA_TAG, MPI_COMM_WORLD, &sreq);
      }

      /* send the class state */
      for (int i = 1; i < numtasks; i++) {
         f.marshall(rbuf);
         MPI_Isend (rbuf, RSIZ, MPI_BYTE, i, CLASS_TAG, MPI_COMM_WORLD, &sreq);
      }

      /* send the function addr */
      for (int i = 1; i < numtasks; i++) {
         sbuf[0] = (size_t)f.rstart;
         MPI_Isend (rbuf, RSIZ, MPI_BYTE, i, FUNC_TAG, MPI_COMM_WORLD, &sreq);
      }

      /* complete the work for the master task */
      int mcount = count - (task_count * (numtasks - 1));
      for (int k = 0; send_it != e; send_it++, k++) {
         buf[k] = *send_it;
      }
      f.rstart((char*)&f,mcount,(char*)buf);

      return;
   }

   void master_terminate() {
      char    buf[RSIZ];
      char   *rbuf = (char*)buf;
      MPI_Request sreq;

      // Stop after the foreach on each of the tasks
      for (int i = 1; i < numtasks; i++) {
         MPI_Isend (rbuf, RSIZ, MPI_BYTE, i, STOP_TAG, MPI_COMM_WORLD, &sreq);
      }

      MPI_Finalize();
      return;
   }


   template<typename IterTy, typename FunctionTy>
   void for_each_task_impl(IterTy b, IterTy e, FunctionTy f) {
      /* Copy the data over to the other tasks */
      typedef typename std::iterator_traits<IterTy>::value_type T;
      master_send_data<IterTy,T,FunctionTy>(b, e, f);

      /* continue communication till all threads terminate */
      task_terminate();

      /* Execute a barrier to sync all the tasks */
      task_synchronize();
   }
}
#endif
