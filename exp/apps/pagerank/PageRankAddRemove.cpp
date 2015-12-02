#include "PageRankDet.h"
#include "PageRankIKDG.h"

int main(int argc, char* argv[]) {
  PageRankIKDG<true> p;

  return p.run(argc, argv);
}
