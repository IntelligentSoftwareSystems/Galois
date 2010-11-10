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
  if (argc != 5) {
    cout << "Usage: sssp <num-threads> <bfs> <maxNode> <input-file>" << endl;
    cout<< "A zero for <num-threads> indicates sequential execution." << endl;
    return -1;
  }

  int threads = atoi(argv[1]);
  bool bfs = strcmp(argv[2], "f") == 0 ? false : true;
  int maxNodes = atoi(argv[3]);
  char* inputfile = argv[4];
  
  SSSP sssp;
  sssp.run(bfs, inputfile, threads, maxNodes);
  return 0;
}
