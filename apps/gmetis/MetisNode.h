/*
 * MetisNode.h
 *
 *  Created on: Jun 13, 2011
 *      Author: xinsui
 */

#ifndef METISNODE_H_
#define METISNODE_H_

typedef int METISINT;
typedef double METISDOUBLE;
#include <stddef.h>
#include <vector>
using namespace std;
class MetisNode{
public:

	MetisNode(int id, int weight) {
		init();
		_id = id;
		_weight = weight;
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
//		_partEd = NULL;
//		_partIndex = NULL;
		_partEd.resize(0);
		_partIndex.resize(0);
		_edgeWgtSum = 0;
		_isBoundary = false;
		_gain = 0;
		_edegree = 0;
		_idegree = 0;
		_ndgrees = 0;
		_numEdges = 0;
		_partition = -1;
	}
//	MetisNode(const MetisNode& node) {
//		_id = node._id;
//		_weight = node._weight;
//		_edgeWgtSum = node._edgeWgtSum;
//	}


//	~MetisNode(){
//		if(_partEd!=NULL){
//			delete[] _partEd;
//			_partEd = NULL;
//		}
//		if(_partIndex!=NULL){
//			delete[] _partIndex;
//			_partIndex = NULL;
//		}
//	}

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

//	void setMapTo(GNode mapTo) {
//		_mapTo = mapTo;
//	}

//	GNode getMapTo() {
//		return _mapTo;
//	}

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



	vector<METISINT>& getPartEd(){
		return _partEd;
	}

	vector<METISINT>& getPartIndex(){
		return _partIndex;
	}

	void initPartEdAndIndex(int num){
//		_partEd = new int[num];
//		_partIndex = new int[num];
		_partEd.resize(num);
		_partIndex.resize(num);
		for(int i=0;i<num;i++){
			_partEd[i] = 0;
			_partIndex[i] = 0;
		}
	}

private:
	METISINT _weight;
	METISINT _numEdges;
	//the sum of weights of its edges
	METISINT _edgeWgtSum;
	METISINT _partition;
	bool _isBoundary;
//	GNode _match;
//	bool _isMatched;
	// the node it maps to in the coarser graph
//	GNode _mapTo;

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
	vector<METISINT> _partEd;
	vector<METISINT> _partIndex;
};

#endif /* METISNODE_H_ */
