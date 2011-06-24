/*
 * GMetisConfig.h
 *
 *  Created on: Jun 13, 2011
 *      Author: xinsui
 */

#ifndef GMETISCONFIG_H_
#define GMETISCONFIG_H_



#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "MetisNode.h"
#include <stdlib.h>
typedef int METISINT;
typedef double METISDOUBLE;

typedef Galois::Graph::FirstGraph<MetisNode,METISINT,true>            GGraph;
typedef Galois::Graph::FirstGraph<MetisNode,METISINT,true>::GraphNode GNode;
#include <set>
using namespace std;
typedef set< GNode > GNodeSet;
template <typename T>
void arrayFill(T* array, int length, T value){
	for(int i=0;i<length;++i){
		array[i] = value;
	}
}

int gNodeToInt(GNode node);
int getRandom(int num);

struct PerCPUValue{
	int mincutInc;
	GNodeSet changedBndNodes;
	PerCPUValue(){
		mincutInc = 0;
	}
};
void merge(PerCPUValue& a, PerCPUValue& b);


#endif /* GMETISCONFIG_H_ */
