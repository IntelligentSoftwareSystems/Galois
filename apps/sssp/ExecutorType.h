/*
 * ExecutorType.h
 *
 *  Created on: Oct 26, 2010
 *      Author: amshali
 */

#ifndef EXECUTORTYPE_H_
#define EXECUTORTYPE_H_

enum ExecutorTypeEnum {SSSP_DEFAULT, BFS_DEFAULT};

class ExecutorType {
private:
public:
	bool bfs;
	ExecutorType(){};
	ExecutorType(bool _bfs) : bfs(_bfs) {};
};

#endif /* EXECUTORTYPE_H_ */
