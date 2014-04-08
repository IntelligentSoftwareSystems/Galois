#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>

#include "smat.h"


using namespace std;


int main(int argc, char* argv[]){

	smat_t Y;
	Y.load_from_binary(argv[1]);
	printf("m %ld n %ld nnz %ld\n", Y.rows, Y.cols, Y.nnz);

	return 0;
}

