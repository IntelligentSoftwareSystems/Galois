#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <stdlib.h>

using namespace std;

int multiple_mains() {
	string filename = "randGraph400x400.txt";
	char str[100];
	string tt;
	int width, height, dummy;
	ifstream scanner;
	scanner.open(filename.c_str());
	scanner >> width;
	scanner >> height;
	scanner >> dummy;

	cout << width << height << dummy << endl;

	scanner.getline(str, 100);

	/*
	 numNodes=width*height;
	 int sourceId=numNodes++;
	 int sinkId=numNodes++;

	 nodes = new GNode[numNodes];
	 for (int i = 0; i < numNodes; i++)
	 {
	 Node *n = new Node(i);
	 if (i == sourceId)
	 n->setSource(numNodes);
	 if (i == sinkId)
	 n->setSink();
	 nodes[i] = graph->createNode(n);
	 graph->add(nodes[i]);
	 }



	 scanner.getline(str,100);
	 cout<<str<<endl;
	 */

	scanner.getline(str, 100);

	if (strstr(str, "N: ") != NULL)
		cout << "this line starts with N: " << endl;

	int k;
	char* small;
	small = strtok(str, " \n");
	for (int i = 0; i < 3; i++) {
		k = atoi(small);
		cout << k << endl;
		small = strtok(NULL, " \n");
	}

	// cout<<str<<endl ;i--;


	scanner.close();
}
