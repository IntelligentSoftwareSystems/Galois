/*
 * Node.cpp
 *
 *  Created on: Aug 30, 2013
 *      Author: kjopek
 */

#include "Node.h"
#include "Production.h"
#include "EProduction.hxx"

#include <Galois/Galois.h>

#include <sys/time.h>
#include <sched.h>

void Node::execute()
{

    productions->Execute(productionToExecute,v,input);
	//struct timeval t1, t2;

	//gettimeofday(&t1, NULL);

	//int tid = galois::Runtime::LL::getTID();

	//gettimeofday(&t2, NULL);

	//printf("Production: %d executed on [%d / %d] in: %f [s]\n",
	//		productionToExecute, tid, galois::Runtime::LL::getPackageForSelf(tid),
	//		((t2.tv_sec-t1.tv_sec) * 1e6 + (t2.tv_usec-t1.tv_usec))/1e6
	//		);

}

