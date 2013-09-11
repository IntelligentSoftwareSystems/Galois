/*
 * Node.cpp
 *
 *  Created on: Aug 30, 2013
 *      Author: kjopek
 */

#include "Node.h"
#include "EProduction.hxx"

#include <Galois/Galois.h>


#include <sys/time.h>
#include <sched.h>

void Node::execute()
{
	//struct timeval t1, t2;

	//gettimeofday(&t1, NULL);

	//int tid = Galois::Runtime::LL::getTID();

	switch (productionToExecute) {
	case A1:
		productions->A1(v, input);
		break;
	case A:
		productions->A(v, input);
		break;
	case AN:
		productions->AN(v, input);
		break;
	case A2NODE:
		productions->A2Node(v);
		break;
	case A2ROOT:
		productions->A2Root(v);
		break;
	case BS:
		productions->BS(v);
		break;
	default:
		printf("Invalid production!\n");
		break;
	}

	//gettimeofday(&t2, NULL);

	//printf("Production: %d executed on [%d / %d] in: %f [s]\n",
	//		productionToExecute, tid, Galois::Runtime::LL::getPackageForSelf(tid),
	//		((t2.tv_sec-t1.tv_sec) * 1e6 + (t2.tv_usec-t1.tv_usec))/1e6
	//		);

}

