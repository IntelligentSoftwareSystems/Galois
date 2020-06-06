#pragma once

#include "component.h"
#include "moderngpu/kernel_reduce.hxx"

struct comp_data {
  Shared<int> weight;
  Shared<int> edge;
  Shared<int> node;
  Shared<int> level;
  Shared<int> dstcomp;

  int* lvl;
  int* minwt;
  int* minedge; // absolute
  int* minnode;
  int* mindstcomp;
};

static void dump_comp_data(struct comp_data comp, int n, int lvl);

static void dump_comp_data(struct comp_data comp, int n, int lvl) {
  int *level, *minwt, *minedge, *minnode, *mindstcomp;

  level      = comp.level.cpu_rd_ptr();
  minwt      = comp.weight.cpu_rd_ptr();
  minedge    = comp.edge.cpu_rd_ptr();
  minnode    = comp.node.cpu_rd_ptr();
  mindstcomp = comp.dstcomp.cpu_rd_ptr();

  for (int i = 0; i < n; i++) {
    if (level[i] == lvl) {
      fprintf(stderr, "%d: (%d) node %d edge %d weight %d dstcomp %d\n", i,
              level[i], minnode[i], minedge[i], minwt[i], mindstcomp[i]);
    }
  }

  comp.level.gpu_wr_ptr();
  comp.weight.gpu_wr_ptr();
  comp.edge.gpu_wr_ptr();
  comp.node.gpu_wr_ptr();
  comp.dstcomp.gpu_wr_ptr();
}
