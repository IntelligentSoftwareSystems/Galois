/*
 * main.cpp
 *
 *  Created on: Oct 26, 2010
 *      Author: amshali
 */

#include "SSSP.h"
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char *argv[]) {
	char* inputfile = NULL;
	bool bfs;
	int threads = 0;
	if (argc != 4) {
		cout << "Usage: sssp <num-threads> <bfs> <input-file>" << endl;
		cout<< "A zero for <num-threads> indicates sequential execution." << endl;
		exit(-1);
	} else {
		threads = atoi(argv[1]);
		bfs = strcmp(argv[2], "f") == 0 ? false : true;
		inputfile = argv[3];
	}

	SSSP sssp = SSSP();
	sssp.run(bfs, inputfile, threads);
	return 0;
}
