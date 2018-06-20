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

//
// dag_lw.h - Implementation of LocalWorklist, a distributed priority queue
//

#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

#include <queue>

template <typename ItemType, ItemType null_value, class CompareType>
struct LocalWorklist {
  typedef std::vector<LocalWorklist> threadlist;
  std::priority_queue<ItemType, std::vector<ItemType>, CompareType> Q;
  pthread_mutex_t lock;
  pthread_cond_t cond;
  threadlist* neighbors;
  unsigned int tid;

  /* Token for termination detection */
  unsigned int token, done_work;

  LocalWorklist(threadlist* neighbors, unsigned int tid, CompareType comparator)
      : Q(comparator), neighbors(neighbors), tid(tid), token(~0), done_work(0) {
    if (pthread_mutex_init(&lock, NULL) || pthread_cond_init(&cond, NULL))
      abort();
  }
  ~LocalWorklist() {
    if (pthread_mutex_destroy(&lock) || pthread_cond_destroy(&cond))
      abort();
  }

  /* Token management */
  void forward_token() {
    /* Lock must already be held */
    if (token == ~(unsigned)0)
      return; /* We don't own the token */
    /* Update token */
    if (done_work) {
      token     = 0;
      done_work = 0;
    } else
      token++;
    /* Forward token to next neighbor */
    unsigned int next = (tid + 1) % neighbors->size();
    if (next != tid) {
      (*neighbors)[next].set_token(token);
      token = ~(unsigned)0;
    }
  }
  void set_token(unsigned int newtoken) {
    if (pthread_mutex_lock(&lock))
      abort();
    token = newtoken;
    if (pthread_cond_broadcast(&cond))
      abort();
    if (pthread_mutex_unlock(&lock))
      abort();
  }
  int process_token() {
    if (token == ~(unsigned)0)
      return 1; /* Keep going */
    return token > neighbors->size() * 2 ? 0 /* We're done */ : 1;
  }

  /* Work stealing */
  void steal() {
    if (neighbors->size() < 2)
      return; /* One thread; can't steal */
    /* Release our lock; we don't need it and we can avoid deadlocks */
    if (pthread_mutex_unlock(&lock))
      abort();
    /* Pick two random neighbors, the second different from the first */
    unsigned one = rand() % neighbors->size();
    unsigned two = rand() % (neighbors->size() - 1);
    if (two >= one)
      two++;
    /* Choose one of the two neighbors to steal from */
    unsigned cone = (*neighbors)[one].peek(), ctwo = (*neighbors)[two].peek();
    unsigned target = cone > ctwo ? one : two;
    unsigned count  = cone > ctwo ? cone : ctwo;
    /* Steal some work from the target */
    if (count > 1) {
      if (count > 5)
        count = count / 4;
      else
        count = 1;
      (*neighbors)[target].transfer(tid, count);
    }
    /* Relock and return */
    if (pthread_mutex_lock(&lock))
      abort();
  }

  /* Push and pop operations (public interface) */
  void push(ItemType node) {
    /* Otherwise push locally */
    if (pthread_mutex_lock(&lock))
      abort();
    Q.push(node);
    if (pthread_mutex_unlock(&lock))
      abort();
    if (pthread_cond_broadcast(&cond))
      abort();
  }
  ItemType pop() {
    if (pthread_mutex_lock(&lock))
      abort();
    while (Q.empty() && process_token()) {
      forward_token();
      steal();
    }
    ItemType node = null_value;
    if (Q.empty()) {
      /* We must be done */
      assert(!done_work);
      forward_token();
    } else {
      /* Pop the top item from the queue and return it */
      node = Q.top();
      Q.pop();
      done_work = 1;
    }
    if (pthread_mutex_unlock(&lock))
      abort();
    return node;
  }
  /* Queue interface for work stealing */
  unsigned transfer(unsigned dest, unsigned count) {
    if (pthread_mutex_lock(&lock))
      abort();
    unsigned n = 0;
    while (!Q.empty() && n < count) {
      ItemType node = Q.top();
      Q.pop();
      (*neighbors)[dest].push(node);
      n++;
    }
    if (pthread_mutex_unlock(&lock))
      abort();
    return n;
  }
  unsigned peek() {
    if (pthread_mutex_lock(&lock))
      abort();
    unsigned count = Q.size();
    if (pthread_mutex_unlock(&lock))
      abort();
    return count;
  }
};
