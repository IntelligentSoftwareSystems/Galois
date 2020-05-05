/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#pragma once

void dump_element(FORD *nodex, FORD *nodey, uint3 &element, int ele)
{
  printf("%f %f %f %f", nodex[element.x], nodey[element.x],
	 nodex[element.y], nodey[element.y]);

  if(!IS_SEGMENT(element))
    printf(" %f %f", nodex[element.z], nodey[element.z]);

  printf(" %d", ele);

}

void dump_neighbours(ShMesh &mesh)
{
  FORD *nodex = mesh.nodex.cpu_rd_ptr();
  FORD *nodey = mesh.nodey.cpu_rd_ptr();
  uint3 *elements = mesh.elements.cpu_rd_ptr();
  uint3 *neighbours = mesh.neighbours.cpu_rd_ptr();

  for(int i = 0; i < mesh.nelements; i++)
    {
      printf("center element [");
      dump_element(nodex, nodey, elements[i], i);
      printf("]\npre-graph\n");
      
      if(neighbours[i].x != INVALIDID) {
	printf("["); dump_element(nodex, nodey, elements[neighbours[i].x], i); printf("]\n");
      }

      if(neighbours[i].y != INVALIDID) {
	printf("["); dump_element(nodex, nodey, elements[neighbours[i].y], i); printf("]\n");
      }

      if(neighbours[i].z != INVALIDID) {
	printf("["); dump_element(nodex, nodey, elements[neighbours[i].z], i); printf("]\n");
      }	
      printf("post-graph\n");
      
      printf("update over\n");      
    }  
}


void debug_isbad(Worklist2 &wl, ShMesh &mesh) {
  bool *isbad = mesh.isbad.cpu_rd_ptr();
  bool *isdel = mesh.isdel.cpu_rd_ptr();

  int i;
  int badcount = 0;

  wl.update_cpu();
  int wlitems = wl.nitems();

  printf("checking %d elements\n", mesh.nelements);
  for(i = 0; i < mesh.nelements; i++) {
    if(isdel[i])
      continue;
    
    if(isbad[i]) {
      badcount++;

      int j = 0;
      for(j = 0; j < wlitems; j++) {
	if(wl.wl[j] == i)
	  break;
      }

      if(j == wlitems)
	printf("\tnot found: %d\n", i);	
    }
  }

  printf("bad count: %d\n", badcount);  
}
