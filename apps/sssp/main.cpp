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
	if (argc != 3) {
		cout << "Usage: sssp <bfs> <input-file>" << endl;
		exit(-1);
	} else {
		bfs = strcmp(argv[1], "f") == 0 ? false : true;
		inputfile = argv[2];
	}

	SSSP sssp = SSSP();
	sssp.run(bfs, inputfile);
	exit(0);
}
