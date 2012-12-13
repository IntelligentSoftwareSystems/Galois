#ifndef GALOIS_RUNTIME_NODEREQUEST_H
#define GALOIS_RUNTIME_NODEREQUEST_H

#include "mpi.h"
#include "pthread.h"
#include <iostream>
#include <set>
#include <unordered_map>
#include <cstring>
#include "Galois/Runtime/Context.h"

#define REQ_TAG  423
#define SERV_TAG 999
#define BUF_SIZE 1000

using namespace std;

namespace Galois {
namespace Runtime {

static bool NR_init = false;

   typedef pair<int,void*> Pair;
   class HashFunction {
     public:
      size_t operator ()(const Pair &element) const {
        return (size_t)element.second;
      }
   };
   class SetEqual {
     public:
      bool operator ()(const Pair &a, const Pair &b) const {
        return ((a.first == b.first) && (a.second == b.second));
      }
   };

   struct locData {
      size_t size;
      int toTask;
   };

class NodeRequest {

protected:
   struct handle_req {
      MPI_Request *mreq;
      size_t  size;
      void   *addr;
      int     req_to;
      int     req_from;
      int     done;
      int     sent;
      int     nodeid;
      int     version;
      bool    ret_req; // set when request for node to be returned
      char    buff[BUF_SIZE];
   };

   struct Requests {
      handle_req  *hreq;
      MPI_Request *mreq;
   };
   typedef struct Requests Requests;

   typedef std::set<handle_req*> req_queue;
   typedef std::set<Requests*>   inreq_que;

   handle_req  **hreq_ser;
   MPI_Request **mreq_ser;

   /* this is the set of all incoming requests that has to be serviced */
   req_queue service_reqs;
   /* the outgoing requests for data from other nodes */
   req_queue outgoing_reqs;
   /* moved from outgoing_reqs when the requested data is received */
   inreq_que incoming_reqs;

   /* set of all initial requests to be serviced after getting remote nodes */
   unordered_map<Pair,handle_req*,HashFunction,SetEqual>       initial_reqs;
   unordered_map<Pair,void*,HashFunction,SetEqual>             recv_hash;
   unordered_map<Pair,locData*,HashFunction,SetEqual>          sent_hash;
   unordered_map<Pair,int,HashFunction,SetEqual>               sent_not_recv_hash;
   unordered_map<Pair,int,HashFunction,SetEqual>               ver_num;
   typedef unordered_map<Pair,void*,HashFunction,SetEqual>     recv_type;
   typedef unordered_map<Pair,locData*,HashFunction,SetEqual>  sent_type;

   /* These variables store the request handles */
   handle_req  *hreq;
   MPI_Request *mreq;

   MPI_Request *allocate_mpi_req()
   {
      MPI_Request *req;

      req = (MPI_Request*) malloc (sizeof(MPI_Request));
      if (!req) {
         printf ("Error could not allocate memory for MPI_Request buffer!\n");
         exit (1);
      }

      return req;
   }

   void free_mpi_req (MPI_Request *req)
   {
      free (req);
   }

   handle_req *allocate_req()
   {
      handle_req *req;

      req = (handle_req*) malloc (sizeof(handle_req));
      if (!req) {
         printf ("Error could not allocate memory for request buffer!\n");
         exit (1);
      }
      req->done = req->sent = 0;

      return req;
   }

   void free_req (handle_req *req)
   {
      free (req);
   }

   Requests *allocate_incoming_req()
   {
      Requests *req;

      req = (Requests*) malloc (sizeof(Requests));
      if (!req) {
         printf ("Error could not allocate memory for incoming buffer!\n");
         exit (1);
      }

      return req;
   }

   void free_incoming_req (Requests *req)
   {
      free (req);
   }

   bool alreadyRequested (void *addr, int task, size_t size) {
      inreq_que::const_iterator pos1;
      req_queue::const_iterator pos2;
      int check;

      // we should let multiple requests for a failsafe system
 /*
      Pair p(task,addr);
      // check if request has been placed but not received
      if (task != taskRank) {
         if (sent_not_recv_hash[p])
           return true;
      }
      else if (sent_hash[p]) {
         p.first = (sent_hash[p])->toTask;
         if (sent_not_recv_hash[p])
           return true;
      }
  */

      for (pos1 = incoming_reqs.begin(); pos1 != incoming_reqs.end(); ++pos1) {
         Requests *woreq = *pos1;
         check = woreq->hreq->req_to;
         if (taskRank == check)
           check = woreq->hreq->req_from;
         if ((task == check) && (addr == woreq->hreq->addr))
            return true;
      }
      for (pos2 = service_reqs.begin(); pos2 != service_reqs.end(); ++pos2) {
         handle_req *woreq = *pos2;
         check = woreq->req_to;
         if (taskRank == check)
           check = woreq->req_from;
         if ((task == check) && (addr == woreq->addr))
            return true;
      }
      for (pos2 = outgoing_reqs.begin(); pos2 != outgoing_reqs.end(); ++pos2) {
         handle_req *woreq = *pos2;
         check = woreq->req_to;
         if (taskRank == check)
           check = woreq->req_from;
         if ((task == check) && (addr == woreq->addr))
            return true;
      }

      return false;
   }

public:
   int numTasks;
   int taskRank;
   pthread_mutex_t mutex;

   void PlaceRequest(int req_to, void *addr, size_t size)
   {
      pthread_mutex_lock(&mutex);
      if (alreadyRequested(addr,req_to,size)) {
         pthread_mutex_unlock(&mutex);
         return;
      }
      handle_req *req = allocate_req();
      req->mreq = allocate_mpi_req();
      req->req_from = taskRank;
      req->req_to = req_to;
      req->addr = addr;
      req->size = size;
      req->version = 0;
      req->ret_req = false;
      if (req_to == taskRank) {
         Pair p(taskRank,addr);
         req->req_to = (sent_hash[p])->toTask;
         req->ret_req = true;
      }
      outgoing_reqs.insert (req);
      pthread_mutex_unlock(&mutex);
      return;
   }

   void *remoteAccess (void *addr, int task)
   {
      void *tmp;
      Pair p(task,addr);
      pthread_mutex_lock(&mutex);
      tmp = recv_hash[p];
      // check for a local node
      if(task == taskRank) {
         // check if the node has been sent to another task
         if (sent_hash[p])
           tmp = NULL;
         else
           tmp = addr;
      }
      // if present lock the node for use
      // only locks if the node isn't already locked
      if (tmp) {
         Lockable *L;
         L = reinterpret_cast<Lockable*>(tmp);
         if (trylock(L))
           setLockValue(L);
      }
      pthread_mutex_unlock(&mutex);
      return tmp;
   }

   void checkRequest (void *addr, int task, void **buf, size_t size)
   {
      inreq_que::const_iterator pos1;

      pthread_mutex_lock(&mutex);
      for (pos1 = incoming_reqs.begin(); pos1 != incoming_reqs.end(); ++pos1) {
         Requests *woreq = *pos1;
         int tchk = woreq->hreq->req_to;
         // if getting back a node
         if ((woreq->hreq->ret_req) && (addr == woreq->hreq->addr)) {
            Pair p(woreq->hreq->req_from,woreq->hreq->addr);
            free(sent_hash[p]);
            sent_hash[p] = NULL;
 /*
            Lockable *L;
            L = reinterpret_cast<Lockable*>(woreq->hreq->addr);
 printf ("\ttask: %d checkRequest unlock!\n", taskRank);
            unlock(L);
  */
            tchk = woreq->hreq->req_from;
         }
         if ((task == tchk) && (addr == woreq->hreq->addr)) {
            if (size > BUF_SIZE) {
               cout << "Error in memory size! " << __FILE__ << ":" << __LINE__ << endl;
               exit(1);
            }
            Pair p(task,addr);
            *buf = recv_hash[p];
            // check if local node
            if (task == taskRank) {
               // update the value in addr
	      std::memcpy(addr,*buf,size);
               free (recv_hash[p]);
               recv_hash[p] = NULL;
               *buf = addr;
            }
            incoming_reqs.erase(woreq);
            free_req(woreq->hreq);
            free_mpi_req(woreq->mreq);
            free_incoming_req(woreq);
            break;
         }
      }
      pthread_mutex_unlock(&mutex);

      return;
   }

   void register_initial_req(int req_from, int req_to, void *addr, size_t size, int ver)
   {
      // when the local node with addr is got back from req_from 
      Pair p(taskRank,addr);
      handle_req *req = allocate_req();
      req->mreq = allocate_mpi_req();
      req->req_from = req_to;
      req->req_to = taskRank;
      req->addr = addr;
      req->size = size;
      req->version = ver;
      req->ret_req = false;
      initial_reqs[p] = req;
      return;
   }

   // requests in initial_reqs moved to service_reqs when received
   void service_initial_reqs()
   {
      inreq_que temp_rem_list;
      inreq_que::const_iterator pos;
      handle_req *req;

      for (pos = incoming_reqs.begin(); pos != incoming_reqs.end(); ++pos) {
         Requests *woreq = *pos;
         Pair p(taskRank,woreq->hreq->addr);
         if ((req = initial_reqs[p])) {
            /* check if the node is marked for lock use */
            initial_reqs[p] = NULL;
            (sent_hash[p])->toTask = req->req_from;
            memcpy(req->buff,woreq->hreq->buff,req->size);
            req->done = 1;
            req->sent = 0;
            service_reqs.insert(req);
            temp_rem_list.insert(woreq);
         }
      }
      // clear all the handled requests
      for (pos = temp_rem_list.begin(); pos != temp_rem_list.end(); ++pos) {
         Requests *woreq = *pos;
         incoming_reqs.erase(woreq);
         free_req(woreq->hreq);
         free_mpi_req(woreq->mreq);
         free_incoming_req(woreq);
      }
      temp_rem_list.clear();
      return;
   }

   void return_remote()
   {
      pthread_mutex_lock(&mutex);
      for (sent_type::iterator it = sent_hash.begin(); it != sent_hash.end(); ++it) {
         if ((*it).second) {
            // PlaceRequest(((*it).second)->toTask,(*it).first.second,((*it).second)->size);
            handle_req *nreq = allocate_req();
            nreq->mreq = allocate_mpi_req();
            nreq->req_from = taskRank;
            nreq->req_to = ((*it).second)->toTask;
            nreq->addr = (*it).first.second;
            nreq->size = ((*it).second)->size;
            nreq->ret_req = true;
            outgoing_reqs.insert (nreq);
         }
      }
      pthread_mutex_unlock(&mutex);
   }

   // temporary service request callback
   void temp_service_req(handle_req *req, req_queue *rem_req)
   {
      // only called from comm() so no mutex required around sent_hash
      locData *loc;
      void    *addr = req->addr;
      Pair     p(req->req_from,req->addr);
      Pair     p1(taskRank,req->addr);
      if ((loc = sent_hash[p1])) {
         // if there is already another request for this node don't place
         if (initial_reqs[p1])
           return;
         // register the initial request to be handled once the node is returned
         register_initial_req(loc->toTask,req->req_from,req->addr,req->size,req->version);
         // place a request to the node to return this node
         handle_req *nreq = allocate_req();
         nreq->mreq = allocate_mpi_req();
         nreq->req_from = taskRank;
         nreq->req_to = loc->toTask;
         nreq->addr = req->addr;
         nreq->size = req->size;
         nreq->version = req->version;
         nreq->ret_req = true;
         outgoing_reqs.insert (nreq);
         // remove this request from the service list
         rem_req->insert(req);
      }
      else {
         // don't service the request if locked
         Lockable *L;
         L = reinterpret_cast<Lockable*>(addr);
         if (!req->ret_req) {
            // this is a local node as not a return request
            // send only if the node is not locked now
            if (!trylock(L))
              return;
         }
         // check if it's a request to return a node
         if(req->ret_req) {
            addr = recv_hash[p];
            L = reinterpret_cast<Lockable*>(addr);
            if (L && getValue(L))
              return;
            recv_hash[p] = NULL;
         }
         // check for possible duplicate requests and delete em
         if (!addr) {
            rem_req->insert(req);
            return;
         }
         memcpy(req->buff,addr,req->size);
         if(req->ret_req) {
            // free the local copy of the remote node
            free(addr);
         }
         if(!req->ret_req) {
            // not a return request so add to the sent_hash
            loc = (locData*)malloc(sizeof(locData));
            loc->toTask = req->req_from;
            loc->size   = req->size;
            sent_hash[p1] = loc;
         }
         req->done = 1;
         req->sent = 0;
      }
      return;
   }

   /* Constructor initializes the MPI environment */
   NodeRequest ()
   {
      int err, count, notasks, rank;
      if (!NR_init) {
         int rc = MPI_Init (NULL, NULL);
         if (rc != MPI_SUCCESS) {
            printf ("Error starting MPI program. Terminating.\n");
            MPI_Abort(MPI_COMM_WORLD, rc);
         }
         pthread_mutex_init(&mutex, NULL);

         // this has to be done only the first time!
         hreq = allocate_req();
         mreq = allocate_mpi_req();

         // check if there are any incoming requests
         err = MPI_Irecv (hreq, sizeof(handle_req), MPI_BYTE, MPI_ANY_SOURCE,
                          REQ_TAG, MPI_COMM_WORLD, mreq);
      }

      MPI_Comm_size(MPI_COMM_WORLD, &notasks);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      numTasks = notasks;
      taskRank = rank;

      if (!NR_init) {
         hreq_ser = (handle_req**)  malloc (sizeof(handle_req*) * numTasks);
         mreq_ser = (MPI_Request**) malloc (sizeof(MPI_Request*) * numTasks);

         // Post a receive for the data for each task
         for (count = 0; count < numTasks; count++) {
            hreq_ser[count] = allocate_req();
            mreq_ser[count] = allocate_mpi_req();
            err = MPI_Irecv (hreq_ser[count], sizeof(handle_req), MPI_BYTE, count,
                             SERV_TAG, MPI_COMM_WORLD, mreq_ser[count]);
         }

         NR_init = true;
      }
   }

   ~NodeRequest()
   {
  //  MPI_Finalize ();
  //  pthread_mutex_destroy(&mutex);
   }

   /*
    * Communicate - This function is called periodically and responsible
    *               for transferring data over the network. The function
    *               services any incoming data requests and also makes data
    *               requests for this thread.
    */
   void Communicate ()
   {
      handle_req  *wreq;
      int          err, flag;
      MPI_Status   status;
      req_queue    temp_add_req, temp_rem_req;
      Requests    *inreq;
      bool         recv_flag;

 pthread_mutex_lock(&mutex);
      /* This do-while loop is used to check for any incoming data requests */
      do {
         /* check the status of the INCOMING message */
         err = MPI_Test (mreq, &flag, &status);
         if (flag) {
            // add a call back to cache manager to service the request
            hreq->mreq = mreq;
            /* add to temp add list so that the request can be serviced */
            temp_add_req.insert (hreq);

            /* try receiving another data request */
            hreq = allocate_req();
            mreq = allocate_mpi_req();
            err = MPI_Irecv (hreq, sizeof(handle_req), MPI_BYTE, MPI_ANY_SOURCE,
                             REQ_TAG, MPI_COMM_WORLD, mreq);
         }
      } while (flag);

      /* loop through the data requests and check if the data can be sent */
      req_queue::const_iterator pos;  
      for (pos = service_reqs.begin(); pos != service_reqs.end(); ++pos) {  
         wreq = *pos;

         /* remove from the list, if the data was already sent */
         if (wreq->done && wreq->sent) {
            err = MPI_Test (wreq->mreq, &flag, &status);
            /* if flag is true the data was sent */
            if (flag) {
               /* add to temp remove list so that it can be removed later */
               temp_rem_req.insert (wreq);
            }
         }

         /* if not serviced yet call service routine */
         if (!wreq->done) {
            temp_service_req (wreq,&temp_rem_req);
         }

         /* check if any data request can be satisfied */
         if (wreq->done && !wreq->sent) {
 //cout << "SERV " << taskRank << " to " << wreq->req_from << " addr " << wreq->addr << endl;
            err = MPI_Isend (wreq, sizeof(handle_req), MPI_BYTE, wreq->req_from,
                             SERV_TAG, MPI_COMM_WORLD, wreq->mreq);
            wreq->sent = 1;
         }
      }

      /* add the new requests to service_reqs */
      for (pos = temp_add_req.begin(); pos != temp_add_req.end(); ++pos) {
         wreq = *pos;
         service_reqs.insert(wreq);
      }

      /* remove all the satisfied requests from service_reqs */
      for (pos = temp_rem_req.begin(); pos != temp_rem_req.end(); ++pos) {
         wreq = *pos;
         service_reqs.erase(wreq);
         free_mpi_req(wreq->mreq);
         free_req(wreq);
      }

      /* empty the temp lists */
      temp_rem_req.erase(temp_rem_req.begin(), temp_rem_req.end());
      temp_add_req.erase(temp_add_req.begin(), temp_add_req.end());

      /* send all the requests made by this node */
      for (pos = outgoing_reqs.begin(); pos != outgoing_reqs.end(); ++pos) {
         wreq = *pos;

         /* send the data request if it hasn't been sent yet */
         if (!wreq->sent) {
 //cout << "REQ " << taskRank << " to " << wreq->req_to << " addr " << wreq->addr << endl;
            // assign version number if not owner
            if (!wreq->ret_req) {
               Pair p(wreq->req_to,wreq->addr);
               wreq->version = ver_num[p];
            }
            err = MPI_Isend (wreq, sizeof(handle_req), MPI_BYTE, wreq->req_to,
                             REQ_TAG, MPI_COMM_WORLD, wreq->mreq);
            wreq->sent = 1;
            Pair p(wreq->req_to,wreq->addr);
            sent_not_recv_hash[p] = 1;
         }
      }

      /* Erase all the entries from the outgoing_reqs queue */
      outgoing_reqs.erase(outgoing_reqs.begin(), outgoing_reqs.end());

      do {
         recv_flag = false;
         /* check if any data has been received from the other tasks */
         for (int task = 0; task < numTasks; task++) {
            MPI_Test (mreq_ser[task], &flag, &status);

            /* flag is true if the request has been sent */
            if (flag) {
               /* add to the incoming_reqs queue */
               inreq = allocate_incoming_req();
               inreq->hreq = hreq_ser[task];
               inreq->mreq = mreq_ser[task];
               incoming_reqs.insert (inreq);

               /* store the incoming message in the recv_hash */
               int storeTask = inreq->hreq->req_to;
               Pair p1(taskRank,inreq->hreq->addr);
               /* check if it's a returned local node */
               if (sent_hash[p1])
                 storeTask = taskRank;
               void *tbuf = (void*)malloc(sizeof(char)*(inreq->hreq->size+5));
               memcpy(tbuf,inreq->hreq->buff,inreq->hreq->size);

               /* lock the data for use by the local task */
               Lockable *L;
               L = reinterpret_cast<Lockable*>(tbuf);
               trylock(L);
               setLockValue(L);
               Pair p2(storeTask,inreq->hreq->addr);
               recv_hash[p2] = tbuf;
               if (storeTask != taskRank) {
                  sent_not_recv_hash[p2] = 0;
                  // unlock if version number doesn't match - node received
                  if (inreq->hreq->version != ver_num[p2]) {
                     unlock(L);
                     // remove from incoming list as it doesn't match to a req
                     incoming_reqs.erase(inreq);
                     free_req(inreq->hreq);
                     free_mpi_req(inreq->mreq);
                     free_incoming_req(inreq);
                  }
                  else
                    ver_num[p2]++;
               }
               else {
                  p2.first = inreq->hreq->req_to;
                  sent_not_recv_hash[p2] = 0;
                  /* lock in directory if sending to remote node */
                  p2.first = taskRank;
                  if (initial_reqs[p2]) {
                     unlock(L);
                     if (!trylock(L)) {
                       cout << "ERROR: " << __FILE__ << ":" << __LINE__ << endl;
                     }
                  }
               }

               /* mark that a request was received */
               recv_flag = true;

               /* place the receive call for the data */
               hreq_ser[task] = allocate_req();
               mreq_ser[task] = allocate_mpi_req();
               err = MPI_Irecv(hreq_ser[task], sizeof(handle_req), MPI_BYTE,
                               task, SERV_TAG, MPI_COMM_WORLD, mreq_ser[task]);
            }
         }
      } while (recv_flag);

      service_initial_reqs();

 pthread_mutex_unlock(&mutex);
      return;
   }

};

} // end namespace
}
#endif
