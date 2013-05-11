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

#ifndef METISNODE_H_
#define METISNODE_H_

typedef int METISINT;
typedef double METISDOUBLE;
#include <stddef.h>
#include <vector>
#include "llvm/ADT/SmallVector.h"
#include <iostream>
#include "Galois/gdeque.h"
using namespace std;
class MetisNode{
public:
	#ifdef NOPARTINFO
	typedef llvm::SmallVector <METISINT,128> svm;
	#else
	typedef int* svm;
	#endif
	/*typedef Galois::gdeque <int*,128> sve;
	sve blah;*/
	int processed;
	MetisNode(int id, int weight) {
		init();
		_id = id;
		_weight = weight;
	//	initPartEdAndIndex(128);
//		_isMatched = false;
	}

	MetisNode(int weight) {
		_id = -1;
		_weight = weight;
		init();
	}

	MetisNode() {
		_weight = 0;
		_id = -1;
		init();
	}

	void init(){
		_edgeWgtSum = 0;
		_isBoundary = false;
		_gain = 0;
		_edegree = 0;
		_idegree = 0;
		_ndgrees = 0;
		_numEdges = 0;
		_partition = -1;
		processed=0;
		matchNode = NULL;
		multiNode = NULL;
		subGraphNode = NULL;
		matched = false;

	}

	int getNodeId() {
		return _id;
	}

	void setNodeId(int i) {
		_id = i;
	}

	int getWeight() {
		return _weight;
	}

	void setWeight(int weight) {
		_weight = weight;
	}

	int getAdjWgtSum() {
		return _edgeWgtSum;
	}

	void addEdgeWeight(int weight) {
		_edgeWgtSum += weight;
	}

	void setAdjWgtSum(int sum) {
		_edgeWgtSum = sum;
	}

	int getPartition() {
		return _partition;
	}

	void setPartition(int part) {
		_partition = part;
	}

	bool isBoundary() {
		return _isBoundary;
	}

	void setBoundary(bool isBoundary) {
		_isBoundary = isBoundary;
	}


	int getIdegree() {
		return _idegree;
	}

	void setIdegree(int idegree) {
		_idegree = idegree;
	}

	int getEdegree() {
		return _edegree;
	}

	void setEdegree(int edegree) {
		_edegree = edegree;
	}

	void swapEDAndID() {
		int temp = _idegree;
		_idegree = _edegree;
		_edegree = temp;
	}

	int getGain() {
		return _gain;
	}

	void updateGain() {
		_gain = _edegree - _idegree;
	}

//	void setSubGraphMap(GNode map) {
//		_subGraphMapTo = map;
//	}
//
//	GNode getSubGraphMap() {
//		return _subGraphMapTo;
//	}

	int getNDegrees() {
		return _ndgrees;
	}

	void setNDegrees(int degrees) {
		_ndgrees = degrees;
	}

	int getNumEdges() {
		return _numEdges;
	}

	void incNumEdges() {
		_numEdges++;
	}

	void incNumEdges(int inc) {
		_numEdges+=inc;
	}

//	METISINT* getPartEd(){
//		return _partEd;
//	}
//
//	METISINT* getPartIndex(){
//		return _partIndex;
//	}

	svm& getPartEd(){
		return _partEd;
	}

	svm& getPartIndex(){
		return _partIndex;
	}

	void initPartEdAndIndex(int nparts){
			for(int i=0;i<nparts;i++) {
				_partEd[i]=0;
				_partIndex[i]=0;
			}

	}


/*
	void * getMatchNode() {
		return matchNode;
	}

	void* getMultiNode() {
		return multiNode;
	}
*/

	bool isMatched() {
		return matched;
	}

	void setMatched(bool matched) {
		this->matched = matched;
	}

	void* getMatchNode() {
		return matchNode;
	}

	void* getMultiNode() {
		return multiNode;
	}


	void* matchNode;
	void *multiNode;
	void *subGraphNode;

private:
	bool matched;
	METISINT _weight;
	METISINT _numEdges;
	//the sum of weights of its edges
	METISINT _edgeWgtSum;
	METISINT _partition;
	bool _isBoundary;

	METISINT _id;
	// the sum of the weights of its edges connecting neighbors in its partition
	METISINT _idegree;
	// the sum of the weights of its edges connecting neighbors in the other partition
	METISINT _edegree;
	// if moving this node to the other partition, the reduced cut
	METISINT _gain;

	// the node it mapped in the subgraph got by bisecting the current graph
	//for kway partitioning
	METISINT _ndgrees;
	svm _partEd;
	svm _partIndex;

};

#endif /* METISNODE_H_ */
