/** GMetis -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @author Xin Sui <xinsui@cs.utexas.edu>
 */

#ifndef PQUEUE_H_
#define PQUEUE_H_
#include "GMetisConfig.h"
#include <algorithm>
#include <vector>
#include <boost/function.hpp>
using namespace std;

template<class T>
struct ListNode{
	ListNode(){
		prev=NULL;
		next=NULL;
	}
	T value;
	ListNode* prev;
	ListNode* next;
};

template<class T>
struct KeyValue {
	int key;
	T value;
};

/**
 * A priority queue which combines two implementations:
 * 1. heap based
 * 2. array(for priority values) + linked-list (saving the elemements of the queue having the same priority)
 * Which one to use depends on the max number of nodes of queue
 */
template <class T>
class SuperPQueue{
public:
	virtual void insert(T value, int priority) = 0;
	virtual void remove(T value, int priority) = 0;
	virtual void update(T value, int oldPriority, int newPriority) = 0;
	virtual	T getMax() = 0;
	virtual	void reset() = 0;
	virtual	int size() = 0;
	virtual ~SuperPQueue(){};
};

template <class T>
class HeapQueue: public SuperPQueue<T>{
public:
	HeapQueue(int maxNumNodes, int maxPriority, boost::function<int(T)> mapToInt){
		numNodes = 0;
		heap.resize(maxNumNodes);
		locator.resize(maxNumNodes); //= new int[maxNumNodes];
		this->mapToInt = mapToInt;
		fill(locator.begin(),locator.end(), -1);
		this->maxNumNodes = maxNumNodes;
	}
	virtual ~HeapQueue(){
	}
	void insert(T value, int priority) {
		//heap
		int i = numNodes;
		numNodes++;
		while (i > 0) {
			int j = (i - 1) / 2;
			if (heap[j].key < priority) {
				heap[i].key = heap[j].key;
				heap[i].value = heap[j].value;
				locator[mapToInt(heap[i].value)] = i;
				i = j;
			} else {
				break;
			}
		}

		heap[i].key = priority;
		heap[i].value = value;
		locator[mapToInt(value)] = i;
	}
	void remove(T value, int priority) {
		int id = mapToInt(value);
		int i = locator[id];
		locator[id] = -1;
		if (--numNodes > 0 && mapToInt(heap[numNodes].value) != id) {
			value = heap[numNodes].value;
			int newPriority = heap[numNodes].key;
			int oldPriority = heap[i].key;

			if (oldPriority < newPriority) {
				while (i > 0) {
					int j = (i - 1) >> 1;
					if (heap[j].key < newPriority) {
						heap[i].value = heap[j].value;
						heap[i].key = heap[j].key;
						locator[mapToInt(heap[i].value)] = i;
						i = j;
					} else
						break;
				}
			} else {
				int j = 0;
				while ((j = 2 * i + 1) < numNodes) {
					if (heap[j].key > newPriority) {
						if (j + 1 < numNodes && heap[j + 1].key > heap[j].key)
							j = j + 1;
						heap[i].value = heap[j].value;
						heap[i].key = heap[j].key;
						locator[mapToInt(heap[i].value)] = i;
						i = j;
					} else if (j + 1 < numNodes && heap[j + 1].key > newPriority) {
						j = j + 1;
						heap[i].value = heap[j].value;
						heap[i].key = heap[j].key;
						locator[mapToInt(heap[i].value)] = i;
						i = j;
					} else
						break;
				}
			}
			heap[i].key = newPriority;
			heap[i].value = value;
			locator[mapToInt(value)] = i;
		}
	}
	void update(T value, int oldPriority, int newPriority) {
		int i = locator[mapToInt(value)];
		if (oldPriority < newPriority) {
			while (i > 0) {
				int j = (i - 1) >> 1;
				if (heap[j].key < newPriority) {
					heap[i].value = heap[j].value;
					heap[i].key = heap[j].key;
					locator[mapToInt(heap[i].value)] = i;
					i = j;
				} else
					break;
			}
		} else {
			int j = 0;
			while ((j = 2 * i + 1) < numNodes) {
				if (heap[j].key > newPriority) {
					if (j + 1 < numNodes && heap[j + 1].key > heap[j].key)
						j = j + 1;
					heap[i].value = heap[j].value;
					heap[i].key = heap[j].key;
					locator[mapToInt(heap[i].value)] = i;
					i = j;
				} else if (j + 1 < numNodes && heap[j + 1].key > newPriority) {
					j = j + 1;
					heap[i].value = heap[j].value;
					heap[i].key = heap[j].key;
					locator[mapToInt(heap[i].value)] = i;
					i = j;
				} else
					break;
			}
		}
		heap[i].key = newPriority;
		heap[i].value = value;
		locator[mapToInt(value)] = i;
	}

	T getMax(){
		if(numNodes == 0){
			assert (false);
			return T();
		}
		numNodes -- ;
		T vtx = heap[0].value;
		locator[mapToInt(vtx)] = -1;
		int i = numNodes;
		if (i > 0) {
			int priority = heap[i].key;
			T node = heap[i].value;
			i = 0;
			int j = 0;
			while ((j = 2 * i + 1) < numNodes) {
				if (heap[j].key > priority) {
					if (j + 1 < numNodes && heap[j + 1].key > heap[j].key)
						j = j + 1;
					heap[i].value = heap[j].value;
					heap[i].key = heap[j].key;
					locator[mapToInt(heap[i].value)] = i;
					i = j;
				} else if (j + 1 < numNodes && heap[j + 1].key > priority) {
					j = j + 1;
					heap[i].value = heap[j].value;
					heap[i].key = heap[j].key;
					locator[mapToInt(heap[i].value)] = i;
					i = j;
				} else
					break;
			}
			heap[i].key = priority;
			heap[i].value = node;
			locator[mapToInt(node)] = i;
		}
		return vtx;
	}
	void reset(){
		numNodes = 0;
//		arrayFill(locator, maxNumNodes, -1);
		fill(locator.begin(),locator.end(), -1);
	}
	int size(){
		return numNodes;
	}

private:
	typedef KeyValue<T> TKV;
	vector<TKV> heap;
	vector<int> locator;
	boost::function<int(T)> mapToInt;
	int numNodes;
	int maxNumNodes;
};

template <class T>
class LimitedPriorityQueue: public SuperPQueue<T>{
public:
	LimitedPriorityQueue(int maxNumNodes, int maxPriority, boost::function<int(T)> mapToInt):PLUS_PRIORITYSPAN(500), NEG_PRIORITYSPAN(500){
		pPrioritySpan = min(PLUS_PRIORITYSPAN, maxPriority);
		nPrioritySpan = min(NEG_PRIORITYSPAN, maxPriority);
		nodes = new ListNode<T>[maxNumNodes];
		bucketSize = pPrioritySpan + nPrioritySpan + 1;
		buckets.resize(bucketSize);
		for(int i=0;i<pPrioritySpan + nPrioritySpan + 1;i++){
			buckets[i] = NULL;
		}
		bucketIndex = nPrioritySpan;
		_maxPriority = 0;
		_maxPriority = -nPrioritySpan;
		_mapToInt = mapToInt;
		numNodes = 0;
	}
	virtual ~LimitedPriorityQueue(){
		delete[] nodes;
	}
	void insert(T value, int priority) {
		numNodes++;
		int id = _mapToInt(value);

		nodes[id].value = value;
		ListNode<T>* newNode = &nodes[id];
		newNode->prev = NULL;
		newNode->next = buckets[priority + bucketIndex];
		if (newNode->next != NULL)
			newNode->next->prev = newNode;

		buckets[priority + bucketIndex] = newNode;
		if (_maxPriority < priority){
			_maxPriority = priority;
		}
	}
	void remove(T value, int priority) {
		numNodes--;
		ListNode<T>* node = &nodes[_mapToInt(value)];
		if (node->prev != NULL)
			node->prev->next = node->next;
		else
			buckets[priority + bucketIndex] = node->next;
		if (node->next != NULL)
			node->next->prev = node->prev;

		if (buckets[priority + bucketIndex] == NULL && priority == _maxPriority) {
			if (numNodes == 0)
				_maxPriority = -nPrioritySpan;
			else
				for (; buckets[_maxPriority + bucketIndex] == NULL; _maxPriority--);
		}
	}
	void update(T value, int oldPriority, int newPriority) {
		remove(value, oldPriority);
		insert(value, newPriority);
	}
	T getMax() {
		if (numNodes == 0){
			cout<<"queue empty..........."<<endl;
			assert (false);
			return T();
		}
		ListNode<T>* tptr = NULL;
		numNodes--;
		tptr = buckets[_maxPriority + bucketIndex];
		buckets[_maxPriority + bucketIndex] = tptr->next;
		if (tptr->next!=NULL) {
			tptr->next->prev = NULL;
		} else {
			if (numNodes == 0) {
				_maxPriority = -nPrioritySpan;
			} else {
				for (; buckets[_maxPriority + bucketIndex] == NULL ; _maxPriority--){
				};
			}
		}
		return tptr->value;
	}

	int size(){
		return numNodes;
	}
	void reset(){
		numNodes = 0;
		_maxPriority = -nPrioritySpan;
//		bucketSize = nPrioritySpan + pPrioritySpan + 1;
		for (int i = 0; i < bucketSize; i++) {
			buckets[i] = NULL;
		}
		bucketIndex = nPrioritySpan;
	}

private:
	int numNodes;
	int _maxPriority;
	int pPrioritySpan;
	int nPrioritySpan;
	ListNode<T>* nodes;
	vector<ListNode<T>*> buckets;
	int bucketSize;
	int bucketIndex;
	const int PLUS_PRIORITYSPAN;
	const int NEG_PRIORITYSPAN;
  boost::function<int(T)> _mapToInt;
};

class PQueue{
public:
  PQueue(int maxNumNodes, int maxGain, GGraph* g) {
    if (maxGain > PLUS_GAINSPAN || maxNumNodes < 500) {
      internalQueue = new HeapQueue<GNode>(maxNumNodes, maxGain, gNodeToInt(g));
    } else {
      internalQueue = new LimitedPriorityQueue<GNode>(maxNumNodes, maxGain, gNodeToInt(g));
    }
  }
  ~PQueue(){
    delete internalQueue;
  }

	/**
	 * insert a node with its gain
	 */
	void insert(GNode node, int gain) {
		internalQueue->insert(node, gain);
	}

	/**
	 * delete a node from the queue
	 */
	void remove(GNode value, int gain) {
		internalQueue->remove(value, gain);
	}

	/**
	 * after changing the gain of a node, its position in the queue has to be updated
	 */
	void update(GNode value, int oldgain, int newgain) {
		internalQueue->update(value, oldgain, newgain);
	}

	/**
	 * return the node with the max gain in the queue
	 */
	GNode getMax() {
		return internalQueue->getMax();
	}

	int size(){
		return internalQueue->size();
	}
	/**
	 * reset the queue
	 */
	void reset() {
		internalQueue->reset();
	}

private:

//	static int (*gNodeToIntPointer)(GNode node) = &(gNodeToInt) ;
	SuperPQueue<GNode>* internalQueue;
	static const int PLUS_GAINSPAN = 500;
};



#endif /* PQUEUE_H_ */
