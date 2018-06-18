#include <iostream>
#include <iomanip>
#include "parallel.h"
#include "geometryIO.h"
#include "geometry.h"
#include "parseCommandLine.h"

int parallel_main(int argc, char* argv[]) {
  Exp::Init iii;
  commandLine P(argc, argv, "<infile> <outFile>");
  pair<char*, char*> fnames = P.IOFileNames();
  char* iFile               = fnames.first;
  char* oFile               = fnames.second;
  _seq<point2d> PIn         = readPointsFromFile<point2d>(iFile);

  ofstream ofile(oFile, ios::out | ios::binary);
  if (!ofile.is_open()) {
    std::cout << "Unable to open file: " << oFile << std::endl;
    return 1;
  }

  ofile << setprecision(11) << PIn.n << " " << 2 << " " << 0 << " " << 0
        << scientific << endl;
  for (int i = 0; i < PIn.n; i++) {
    ofile << i << " " << PIn.A[i].x << " " << PIn.A[i].y << " " << 0 << endl;
  }
  ofile.close();
}
