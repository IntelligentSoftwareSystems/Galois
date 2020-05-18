/* -*- mode: c++ -*- */
#include "meshfiles.h"
#include "dmr.h"
#include "geomprim.h"
#include "gbar.cuh"
#include <cub/cub.cuh>
#include "devel.h"
#include <map>

#undef TB_SIZE
#define TB_SIZE 512

#define LOADCV(x) cub::ThreadLoad<cub::LOAD_CV>((x))
#define LOADCG(x) cub::ThreadLoad<cub::LOAD_CG>((x))
#define STORECG(x, y) cub::ThreadStore<cub::STORE_CG>((x), (y))

void addneighbour_cpu(uint3 &neigh, uint elem)
{
  // TODO
  if(neigh.x == elem || neigh.y == elem || neigh.z == elem) return;

  assert(neigh.x == INVALIDID || neigh.y == INVALIDID || neigh.z == INVALIDID);

  if(neigh.x == INVALIDID) { neigh.x = elem; return; }
  if(neigh.y == INVALIDID) { neigh.y = elem; return; }
  if(neigh.z == INVALIDID) { neigh.z = elem; return; }
}

void find_neighbours_cpu(ShMesh &mesh)
{
  std::map<std::pair<int, int>, int> edge_map;

  uint nodes1[3];

  uint3 *elements = mesh.elements.cpu_rd_ptr();
  uint3 *neighbours = mesh.neighbours.cpu_wr_ptr(true);
  int ele;

  for(ele = 0; ele < mesh.nelements; ele++)
    {
      uint3 *neigh = &neighbours[ele];

      neigh->x = INVALIDID;
      neigh->y = INVALIDID;
      neigh->z = INVALIDID;

      nodes1[0] = elements[ele].x;
      nodes1[1] = elements[ele].y;
      nodes1[2] = elements[ele].z;

      if(nodes1[0] > nodes1[1]) std::swap(nodes1[0], nodes1[1]);
      if(nodes1[1] > nodes1[2]) std::swap(nodes1[1], nodes1[2]);
      if(nodes1[0] > nodes1[1]) std::swap(nodes1[0], nodes1[1]);

      assert(nodes1[0] <= nodes1[1] && nodes1[1] <= nodes1[2]);

      std::pair<int, int> edges[3];
      edges[0] = std::make_pair<int, int>(nodes1[0], nodes1[1]);
      edges[1] = std::make_pair<int, int>(nodes1[1], nodes1[2]);
      edges[2] = std::make_pair<int, int>(nodes1[0], nodes1[2]);

      int maxn = IS_SEGMENT(elements[ele]) ? 1 : 3;
      
      for(int i = 0; i < maxn; i++) {
	if(edge_map.find(edges[i]) == edge_map.end())
	  edge_map[edges[i]] = ele;
	else {	  
	  int node = edge_map[edges[i]];
	  addneighbour_cpu(neighbours[node], ele);
	  addneighbour_cpu(neighbours[ele], node);
	  edge_map.erase(edges[i]);
	}
      }
    }
}

void verify_mesh(ShMesh &mesh)
{
  // code moved to refine_mesh (final invocation of check_triangles)
  // TODO: check for delaunay property
}

void read_mesh(const char *basefile, ShMesh &mesh, int maxfactor)
{
  readNodes(basefile, mesh, maxfactor);
  readTriangles(basefile, mesh, maxfactor);

  assert(mesh.maxnelements > 0);
  printf("memory for owners: %d MB\n", mesh.maxnelements * sizeof(int) / 1048576);
  mesh.owners.alloc(mesh.maxnelements);
  // see refine() for actual allocation
  printf("memory for worklists: %d MB\n", 2 * mesh.nelements * sizeof(int) / 1048576);

  printf("%s: %d nodes, %d triangles, %d segments read\n", basefile, mesh.nnodes, mesh.ntriangles, mesh.nsegments);
  assert(mesh.nnodes > 0);
  assert(mesh.ntriangles > 0);
  assert(mesh.nsegments > 0);
  assert(mesh.nelements > 0);
}

__device__ void check_is_bad(Mesh &mesh, int ele)
{
  uint3 *el = &mesh.elements[ele];

  mesh.isbad[ele] = (angleLT(mesh, el->x, el->y, el->z) 
		     || angleLT(mesh, el->z, el->x, el->y) 
		     || angleLT(mesh, el->y, el->z, el->x));
}

__device__ bool shares_edge(uint nodes1[3], uint nodes2[3])
{
  int i;
  int match = 0;
  uint help;

  for (i = 0; i < 3; i++) {
    if ((help = nodes1[i]) != INVALIDID) {
      if (help == nodes2[0]) match++;
      else if (help == nodes2[1]) match++;
      else if (help == nodes2[2]) match++;
    }
  } 
  // for(i = 0; i < 3; i++)
  //   for(int j = 0; j < 3; j++)
  //     {
  // 	if(nodes1[i] == nodes2[j] && nodes1[i] != INVALIDID)
  // 	  {
  // 	    match++;
  // 	    break;
  // 	  }
  //     }



  return match == 2;
}

__global__ void find_neighbours(Mesh mesh, int start, int end)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int threads = blockDim.x * gridDim.x;
  int ele;
  int oele;
  int nc = 0;
  uint nodes1[3], nodes2[3], neigh[3] = {INVALIDID, INVALIDID, INVALIDID};

  for(int x = 0; x < mesh.nelements; x += 4096) {
    // currently a n^2 algorithm -- launch times out for 250k.ele!
    for(ele = id + start; ele < end; ele += threads)
      {
	if(x == 0)
	  {
	    neigh[0] = INVALIDID;
	    neigh[1] = INVALIDID;
	    neigh[2] = INVALIDID;
	  }
	else
	  {
	    neigh[0] = mesh.neighbours[ele].x;
	    neigh[1] = mesh.neighbours[ele].y;
	    neigh[2] = mesh.neighbours[ele].z;
	  }

	if(neigh[2] != INVALIDID) continue;

	//TODO: possibly remove uint3 from Mesh/ShMesh
	nodes1[0] = mesh.elements[ele].x;
	nodes1[1] = mesh.elements[ele].y;
	nodes1[2] = mesh.elements[ele].z;
	nc = (neigh[0] == INVALIDID) ? 0 : ((neigh[1] == INVALIDID) ? 1 : 2);

	//TODO: block this
	for(oele = 0; oele < mesh.nelements; oele++)
	  {
	    nodes2[0] = mesh.elements[oele].x; 
	    nodes2[1] = mesh.elements[oele].y; 
	    nodes2[2] = mesh.elements[oele].z;

	    if(shares_edge(nodes1, nodes2))
	      {
		assert(nc < 3);
		neigh[nc++] = oele;
	      }

	    if((IS_SEGMENT(mesh.elements[ele]) && nc == 2) || nc == 3)
	      break;
	  }

	mesh.neighbours[ele].x = neigh[0];
	mesh.neighbours[ele].y = neigh[1];
	mesh.neighbours[ele].z = neigh[2];
      }
  }
}

__device__ void dump_mesh_element(Mesh &mesh, uint3 &ele, int element)
{
  if(IS_SEGMENT(ele))
    printf("[ %.17f %.17f %.17f %.17f %d]\n", 
	   mesh.nodex[ele.x], mesh.nodey[ele.x],
	   mesh.nodex[ele.y], mesh.nodey[ele.y], element);
  else
    printf("[ %.17f %.17f %.17f %.17f %.17f %.17f %d]\n", 
	   mesh.nodex[ele.x], mesh.nodey[ele.x],
	   mesh.nodex[ele.y], mesh.nodey[ele.y],
	   mesh.nodex[ele.z], mesh.nodey[ele.z], element);
}
__device__ void dump_mesh_element(Mesh &mesh, int element)
{
  dump_mesh_element(mesh, mesh.elements[element], element);
}

__device__ bool encroached(Mesh &mesh, int element, uint3 &celement, FORD centerx, FORD centery, bool &is_seg)
{
  if(element == INVALIDID)
    return false;

  assert(!mesh.isdel[element]);

  uint3 ele = LOADCG(&mesh.elements[element]);

  if(IS_SEGMENT(ele)) {
    //if(IS_SEGMENT(celement)) //TODO: regd second segment encroaching?
    //  return false;

    FORD cx, cy, radsqr;
    uint nsp;

    is_seg = true;

    nsp = (celement.x == ele.x) ? ((celement.y == ele.y) ? celement.z : celement.y) : celement.x;

    // check if center and triangle are on opposite sides of segment
    // one of the ccws does not return zero
    if(counterclockwise(mesh.nodex[ele.x], mesh.nodey[ele.x], 
    			mesh.nodex[ele.y], mesh.nodey[ele.y],
    			mesh.nodex[nsp], mesh.nodey[nsp]) > 0 != 
       counterclockwise(mesh.nodex[ele.x], mesh.nodey[ele.x], 
    			mesh.nodex[ele.y], mesh.nodey[ele.y],
    			centerx, centery) > 0)
      return true; 

    // nope, do a distance check
    cx = (mesh.nodex[ele.x] + mesh.nodex[ele.y]) / 2;
    cy = (mesh.nodey[ele.x] + mesh.nodey[ele.y]) / 2;
    radsqr = distanceSquare(cx, cy, mesh.nodex[ele.x], mesh.nodey[ele.x]);

    return distanceSquare(centerx, centery, cx, cy) < radsqr;
  } else
    return gincircle(mesh.nodex[ele.x], mesh.nodey[ele.x],
		     mesh.nodex[ele.y], mesh.nodey[ele.y],
		     mesh.nodex[ele.z], mesh.nodey[ele.z],
		     centerx, centery) > 0.0;
}

__device__ void add_to_cavity(uint cavity[], uint &cavlen, int element)
{
  int i;
  for(i = 0; i < cavlen; i++)
    if(cavity[i] == element)
      return;

  cavity[cavlen++] = element;
}

__device__ void add_to_boundary(uint boundary[], uint &boundarylen, uint sn1, uint sn2, uint src, uint dst)
{
  int i;
  for(i = 0; i < boundarylen; i+=4)
    if((sn1 == boundary[i] && sn2 == boundary[i+1]) ||
       (sn1 == boundary[i+1] && sn2 == boundary[i]))
      return;

  boundary[boundarylen++] = sn1;  
  boundary[boundarylen++] = sn2;
  boundary[boundarylen++] = src;
  boundary[boundarylen++] = dst;
}

__device__ unsigned add_node(Mesh &mesh, FORD x, FORD y, uint ndx)
{
  //uint ndx = atomicAdd(&mesh.nnodes, 1);
  assert(ndx < mesh.maxnnodes);

  mesh.nodex[ndx] = x;
  mesh.nodey[ndx] = y;  

  return ndx;
}

__device__ uint add_segment(Mesh &mesh, uint n1, uint n2, uint ndx)
{
  //TODO: parallelize
  uint3 ele;
  ele.x = n1; ele.y = n2; ele.z = INVALIDID;

  //uint ndx = atomicAdd(&mesh.nelements, 1);
  assert(ndx < mesh.maxnelements);

  mesh.isbad[ndx] = false;
  mesh.isdel[ndx] = false;
  mesh.elements[ndx] = ele;
  mesh.neighbours[ndx].x = mesh.neighbours[ndx].y = mesh.neighbours[ndx].z = INVALIDID;

  return ndx;
}

__device__ uint add_triangle(Mesh &mesh, uint n1, uint n2, uint n3, uint nb1, uint oldt, uint ndx)
{
  uint3 ele;
  if(counterclockwise(mesh.nodex[n1], mesh.nodey[n1], 
		      mesh.nodex[n2], mesh.nodey[n2],
		      mesh.nodex[n3], mesh.nodey[n3]) > 0)
    {
      ele.x = n1; ele.y = n2; ele.z = n3;
    }
  else
    {
      ele.x = n3; ele.y = n2; ele.z = n1;
    }

  //uint ndx = atomicAdd(&mesh.nelements, 1);
  assert(ndx < mesh.maxnelements);

  mesh.isbad[ndx] = false;
  mesh.isdel[ndx] = false;
  mesh.elements[ndx] = ele;
  mesh.neighbours[ndx].x = nb1;
  mesh.neighbours[ndx].y = mesh.neighbours[ndx].z = INVALIDID;
  //check_is_bad(mesh, ndx);

  uint3 *nb = &mesh.neighbours[nb1];

  if(mesh.neighbours[nb1].x == oldt)
    nb->x = ndx;
  else {
    if(mesh.neighbours[nb1].y == oldt)
      nb->y = ndx;
    else
      {
	if(mesh.neighbours[nb1].z != oldt)
	  printf("%u %u %u %u %u %u\n", ndx, oldt, nb1, mesh.neighbours[nb1].x, 
		 mesh.neighbours[nb1].y, mesh.neighbours[nb1].z);

	assert(mesh.neighbours[nb1].z == oldt);
	nb->z = ndx;
      }
  }

  // if(mesh.neighbours[nb1].x == oldt)
  //   cub::ThreadStore<cub::STORE_CG>(&mesh.neighbours[nb1].x, ndx);
  // else {
  //   if(mesh.neighbours[nb1].y == oldt)
  //     cub::ThreadStore<cub::STORE_CG>(&mesh.neighbours[nb1].y, ndx);
  //   else
  //     {
  // 	assert(mesh.neighbours[nb1].z == oldt);
  // 	cub::ThreadStore<cub::STORE_CG>(&mesh.neighbours[nb1].z, ndx);
  //     }
  // }

  return ndx;
}

__device__ bool adjacent(uint3 &elem1, uint3 &elem2)
{
  int sc = 0;
  if(elem1.x == elem2.x || elem1.x == elem2.y || elem1.x == elem2.z)
    sc++;

  if(elem1.y == elem2.x || elem1.y == elem2.y || elem1.y == elem2.z)
    sc++;

  if(!IS_SEGMENT(elem1) && (elem1.z == elem2.x || elem1.z == elem2.y || elem1.z == elem2.z))
    sc++;

  return sc == 2;
}
__device__ void find_shared_edge(uint3 &elem1, uint3 &elem2, uint se[2])
{
  int sc = 0;
  if(elem1.x == elem2.x || elem1.x == elem2.y || elem1.x == elem2.z)
    se[sc++] = elem1.x;

  if(elem1.y == elem2.x || elem1.y == elem2.y || elem1.y == elem2.z)
    se[sc++] = elem1.y;

  if(!IS_SEGMENT(elem1) && (elem1.z == elem2.x || elem1.z == elem2.y || elem1.z == elem2.z))
    se[sc++] = elem1.z;

  assert(sc == 2);
  assert(se[0] != INVALIDID);
  assert(se[1] != INVALIDID);
}

__device__ bool build_cavity(Mesh &mesh, uint cavity[], uint &cavlen, int max_cavity, uint boundary[], uint &boundarylen, FORD &cx, FORD &cy)
{
  int ce = 0;
  //FORD cx, cy;
  uint3 ele = LOADCG(&mesh.elements[cavity[0]]);
  bool is_seg = false;

  if(IS_SEGMENT(ele))
    {
      cx = (mesh.nodex[ele.x] + mesh.nodex[ele.y]) / 2;
      cy = (mesh.nodey[ele.x] + mesh.nodey[ele.y]) / 2;
    }
  else
    {
      circumcenter(mesh.nodex[ele.x], mesh.nodey[ele.x],
		   mesh.nodex[ele.y], mesh.nodey[ele.y],
		   mesh.nodex[ele.z], mesh.nodey[ele.z],
		   cx, cy);
    }

  if(debug) printf("highlight %d %d [%f %f]\n", cavity[0], IS_SEGMENT(ele), cx, cy);
  while (ce < cavlen) {
    if(mesh.isdel[cavity[ce]])
      printf("deleted: %d\n", cavity[ce]);

    assert(cavlen < max_cavity);
    assert(!mesh.isdel[cavity[ce]]);


    uint3 neighbours = LOADCG(&mesh.neighbours[cavity[ce]]);
    uint neighb[3] = {neighbours.x, neighbours.y, neighbours.z};

    for(int i = 0; i < 3; i++) {
      if(neighb[i] == cavity[0])
	continue;

      if(neighb[i] == INVALIDID)
	continue;

      //printf("neigbour %d\n", neighb[i]);

      is_seg  = false;
      if(!(IS_SEGMENT(ele) && IS_SEGMENT(LOADCG(&mesh.elements[neighb[i]]))) && 
	 encroached(mesh, neighb[i], ele, cx, cy, is_seg)) {
	if(!is_seg)
	  add_to_cavity(cavity, cavlen, neighb[i]);
	else {
	  assert(!IS_SEGMENT(ele));
	  cavity[0] = neighb[i];
	  cavlen = 1;
	  boundarylen = 0;
	  return false;
	}
      } else {
	uint se[2];
	if(!adjacent(mesh.elements[cavity[ce]], mesh.elements[neighb[i]]))
	  {
	    dump_mesh_element(mesh, cavity[ce]);
	    dump_mesh_element(mesh, neighb[i]);
	    printf("%d %d\n", cavity[ce], neighb[i]);
	  }

	assert(boundarylen < BCLEN);
	find_shared_edge(mesh.elements[cavity[ce]], mesh.elements[neighb[i]], se);
	add_to_boundary(boundary, boundarylen, se[0], se[1], neighb[i], cavity[ce]);
      }
    }
    ce++;
  }

  return true;
}

__device__ void addneighbour(Mesh &mesh, uint3 &neigh, uint elem)
{
  // TODO
  if(neigh.x == elem || neigh.y == elem || neigh.z == elem) return;

  assert(neigh.x == INVALIDID || neigh.y == INVALIDID || neigh.z == INVALIDID);

  if(neigh.x == INVALIDID) { neigh.x = elem; return; }
  if(neigh.y == INVALIDID) { neigh.y = elem; return; }
  if(neigh.z == INVALIDID) { neigh.z = elem; return; }
}

__device__ void setup_neighbours(Mesh &mesh, uint start, uint end)
{
  // relies on all neighbours being in start--end
  for(uint i = start; i < end; i++) {
    uint3 &neigh = mesh.neighbours[i];

    for(uint j = i+1; j < end; j++) {
      if(adjacent(mesh.elements[i], mesh.elements[j]))
	{
	  addneighbour(mesh, neigh, j);
	  addneighbour(mesh, mesh.neighbours[j], i);
	}
    }    
  }
}

__device__ uint opposite(Mesh &mesh, uint element)
{
  bool obtuse = false;
  int obNode = INVALIDID;
  uint3 el = mesh.elements[element];

  if(IS_SEGMENT(el))
    return element;

  // figure out obtuse node
  if(angleOB(mesh, el.x, el.y, el.z)) {
    obtuse = true;
    obNode = el.z;
  } else {
    if(angleOB(mesh, el.z, el.x, el.y)) {
      obtuse = true;
      obNode = el.y;
    } else {
      if(angleOB(mesh, el.y, el.z, el.x)) {
	obtuse = true;
	obNode = el.x;
      }
    }
  }

  if(obtuse) {
    // find the neighbour that shares an edge whose points do not include obNode
    uint se_nodes[2];
    uint nobneigh;

    uint3 neigh = mesh.neighbours[element];

    if(debug) printf("obtuse node [%f %f]\n", mesh.nodex[obNode], mesh.nodey[obNode]);
    assert(neigh.x != INVALIDID && neigh.y != INVALIDID && neigh.z != INVALIDID);

    nobneigh = neigh.x;
    find_shared_edge(el, mesh.elements[neigh.x], se_nodes);
    if(se_nodes[0] == obNode || se_nodes[1] == obNode) {
      nobneigh = neigh.y;
      find_shared_edge(el, mesh.elements[neigh.y], se_nodes);
      if(se_nodes[0] == obNode || se_nodes[1] == obNode) {
	nobneigh = neigh.z;
      }
    }

    return nobneigh;
  }

  return element;
}


__global__
void refine_orig(Mesh mesh, int debg, uint *nnodes, uint *nelements, Worklist2 wl, Worklist2 owl, ExclusiveLocks _ex, GlobalBarrier gb)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int threads = blockDim.x * gridDim.x;
  int ele, eleit, haselem;
  //int debg = 32;
  uint cavity[CAVLEN], nc = 0; // for now
  uint boundary[BCLEN], bc = 0;
  uint ulimit = ((*wl.dindex + threads - 1) / threads) * threads;
  bool repush = false;
  //typedef cub::BlockScan<int, REFINE_BLKSIZE> BlockScan;
  const int perthread = ulimit / threads;
  int stage = 0;
  int x = 0;

  for(eleit = id * perthread; eleit < (id * perthread + perthread) && eleit < ulimit; eleit++, x++)
    {
      haselem = wl.pop_id(eleit, ele);

      //printf("%d:%d:%d:%d\n", x, id, eleit, ele);
      FORD cx, cy;
      nc = 0;
      bc = 0;
      repush = false;
      stage = 0;

      if(haselem && ele < mesh.nelements && mesh.isbad[ele] && !mesh.isdel[ele])
	{
	  cavity[nc++] = ele;

	  if(debug) {
	    printf("original center element ");
	    dump_mesh_element(mesh, cavity[0]);
	  }

	  uint oldcav;
	  do {
	    oldcav = cavity[0];
	    cavity[0] = opposite(mesh, ele);
	  } while(cavity[0] != oldcav);

	  if(!build_cavity(mesh, cavity, nc, CAVLEN, boundary, bc, cx, cy))
	    build_cavity(mesh, cavity, nc, CAVLEN, boundary, bc, cx, cy);

	  if(debug) {
	    printf("center element [%f %f] ", cx, cy);
	    dump_mesh_element(mesh, cavity[0]);
	    printf("pre-graph %d\n", nc);
	    for(int i = 1; i < nc; i++)
	      {
		dump_mesh_element(mesh, cavity[i]);
	      }
	    printf("boundary-edges %d\n", bc);
	    for(int i = 0; i < bc; i+=4) {	
	      printf("[%f %f %f %f]\n", mesh.nodex[boundary[i]], mesh.nodey[boundary[i]],
		     mesh.nodex[boundary[i+1]], mesh.nodey[boundary[i+1]]);
	      dump_mesh_element(mesh, boundary[i+2]);
	    }
	  }

	  // try to claim ownership
	  for(int i = 0; i < nc; i++)
	    STORECG(&mesh.owners[cavity[i]], id);

	  for(int i = 0; i < bc; i+=4)
	    STORECG(&mesh.owners[boundary[i + 2]], id);

	  stage = 1;
	}

      gb.Sync();

      if(stage == 1)
	{
	  // check for conflicts
	  for(int i = 0; i < nc; i++) {
	    if(LOADCG(&mesh.owners[cavity[i]]) != id)
	      atomicMin((int *) &mesh.owners[cavity[i]], id);
	  }

	  for(int i = 0; i < bc; i+=4) {
	    if(LOADCG(&mesh.owners[boundary[i + 2]]) != id)
	      atomicMin((int *) &mesh.owners[boundary[i + 2]], id);
	  }

	  stage = 2;

	}

      gb.Sync();

      int nodes_added = 0;
      int elems_added = 0;
      if(stage == 2)
	{
	  int i;
	  for(i = 0; i < nc; i++)
	    if(LOADCG(&mesh.owners[cavity[i]]) != id) {
	      repush = true;
	      if(debug) printf("%d conflict\n", ele);
	      //printf("%d: %d owned by %d\n", id, cavity[i], mesh.owners[cavity[i]]);
	      break;
	    }

	  if(!repush)
	    for(i = 0; i < bc; i+=4)
	      if(LOADCG(&mesh.owners[boundary[i + 2]]) != id) {
		repush = true;
		if(debug) printf("%d conflict\n", ele);
		//printf("%d: %d owned by %d\n", id, boundary[i + 2], mesh.owners[boundary[i + 2]]);
		break;
	      }

	  // if(!repush)
	  //   {
	  //     for(int i = 0; i < nc; i++)
	  // 	printf("%d:%d:%d\n", x, id, cavity[i]);

	  //     for(int i = 0; i < bc; i+=4)
	  // 	printf("%d:%d:%d\n", x, id, boundary[i + 2]);
	  //   }

	  if(!repush)
	    {
	      stage = 3;

	      nodes_added = 1;
	      elems_added = (bc >> 2) + (IS_SEGMENT(mesh.elements[cavity[0]]) ? 2 : 0);
	    }
	}

      // __syncthreads();
      // typedef cub::WarpScan<int, 1, 32> WarpScan;
      // __shared__ typename WarpScan::TempStorage temp_storage;
      // int total = 0, offset = elems_added;
      // __shared__ int start;
      // WarpScan(temp_storage).ExclusiveSum(elems_added, offset, total);
      // __syncthreads();
      // if((threadIdx.x & 31) == 0 && total > 0) {
      // 	 start = atomicAdd(nelements, total);
      // }
      // __syncthreads();
      // if(total > 0)
      // 	 printf("total: %d %d %d\n", threadIdx.x, id, total);

      // if(elems_added > 0)
      // 	 printf("ea: %d %d\n", id, elems_added);

      if(stage == 3)
	{
	  uint cnode = add_node(mesh, cx, cy, atomicAdd(nnodes, 1));
	  uint cseg1 = 0, cseg2 = 0;

	  uint nelements_added = elems_added;
	  //printf("start: %d %d %d %d %d\n", id, elems_added, start, offset, start+offset);
	  uint oldelements = atomicAdd(nelements, nelements_added);

	  uint newelemndx = oldelements;
	  if(debug) printf("post-graph\n");
	  if(IS_SEGMENT(mesh.elements[cavity[0]]))
	    {
	      cseg1 = add_segment(mesh, mesh.elements[cavity[0]].x, cnode, newelemndx++);
	      cseg2 = add_segment(mesh, cnode, mesh.elements[cavity[0]].y, newelemndx++);
	      if(debug) {
		dump_mesh_element(mesh, cseg1);
		dump_mesh_element(mesh, cseg2);
	      }
	    }

	  for(int i = 0; i < bc; i+=4) {
	    uint ntri  = add_triangle(mesh, boundary[i], boundary[i+1], cnode, boundary[i+2], boundary[i+3], 
				      newelemndx++);
	    //if(mesh.isbad[ntri])
	    //{
	    //printf("puhsing %d\n", ntri);
	    //owl.push(ntri);
	    //}

	    //printf("%d wrote %d\n", id, ntri);
	    if(debug) dump_mesh_element(mesh, ntri);
	  }

	  assert(oldelements + nelements_added == newelemndx);

	  setup_neighbours(mesh, oldelements, newelemndx);

	  repush = true;
	  for(int i = 0; i < nc; i++)
	    {
	      mesh.isdel[cavity[i]] = true;
	      // if the resulting cavity does not contain the original triangle
	      // (because of the opposite() routine, add it back.
	      if(cavity[i] == ele) repush = false;  
	      //printf("%d: deleting %d\n", id, cavity[i]);
	    }

	  if(debug) printf("update over\n");
	  //if(debg-- == 0) break;      
	}

      //owl.push_1item<BlockScan>((repush ? 1 : 0), ele, 512);
      if(repush) owl.push(ele);
      gb.Sync();
    }
}

__global__
void check_triangles_orig(Mesh mesh, unsigned int * bad_triangles, int start, Worklist2 in_wl, Worklist2 wl)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int threads = blockDim.x * gridDim.x;
  int ele;
  uint3 *el;
  int count = 0;
  int ulimit = mesh.nelements; //start + ((mesh.nelements - start + blockDim.x - 1) / blockDim.x) * blockDim.x;
  bool push;

  if(debug && id == 0)
    printf("start %d nelements %d\n", start, mesh.nelements);

  for(ele = id + start; ele < ulimit; ele += threads)
    {
      push = false;

      if(ele < mesh.nelements) {
	if(mesh.isdel[ele])
	  goto next;

	if(IS_SEGMENT(mesh.elements[ele]))
	  goto next;

	if(!mesh.isbad[ele])
	  {
	    el = &mesh.elements[ele];
	    
	    mesh.isbad[ele] = (angleLT(mesh, el->x, el->y, el->z) 
			       || angleLT(mesh, el->z, el->x, el->y) 
			       || angleLT(mesh, el->y, el->z, el->x));
	  }

	if(mesh.isbad[ele])
	  {
	    push = true;
	    count++;
	  }
      }

    next:
      //wl.push_1item<BlockScan>(push ? 1: 0, ele, 384); // slower than push?
      if(push) wl.push(ele); // not really as bad as it looks
    }

  //TODO: replace with warp wide and then block wide
  atomicAdd(bad_triangles, count);
}

void refine_mesh_orig(ShMesh &mesh, dim3 &blocks, dim3 &threads)
{
  ExclusiveLocks refine_ex_locks(mesh.maxnelements);
  Shared<uint> nbad(1);
  Mesh gmesh(mesh);
  Shared<uint> nelements(1), nnodes(1);
  int cnbad;
  GlobalBarrierLifetime gb;
  Worklist2 wl1(mesh.nelements), wl2(mesh.nelements);
  const int REFINE_BLKSIZE = threads.x;
  const size_t RES_REFINE = maximum_residency(refine_orig, REFINE_BLKSIZE, 0); 
  const int nSM = 14;

  if(debug) printf("Running at residency %d.\n", RES_REFINE);

  find_neighbours_cpu(mesh);
  //dump_neighbours(mesh);
  gmesh.refresh(mesh);

  gb.Setup(nSM * RES_REFINE);

  *(nelements.cpu_wr_ptr(true)) = mesh.nelements;
  *(nnodes.cpu_wr_ptr(true)) = mesh.nnodes;

  //double starttime, endtime;
  int lastnelements = 0;
  Worklist2 *inwl = &wl1, *outwl = &wl2;

  *(nbad.cpu_wr_ptr(true)) = 0;
  printf("checking triangles...\n");
  check_triangles_orig<<<nSM, threads.x>>>(gmesh, nbad.gpu_wr_ptr(), 0, *outwl, *inwl); 
  cnbad = *(nbad.cpu_rd_ptr());
  printf("%d initial bad triangles\n", cnbad);

  while(cnbad) {   
    if(debug) inwl->display_items();
    lastnelements = gmesh.nelements;

    refine_orig<<<nSM * RES_REFINE, REFINE_BLKSIZE>>>(gmesh, 32, nnodes.gpu_wr_ptr(), nelements.gpu_wr_ptr(), *inwl, *outwl, refine_ex_locks, gb);
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); // not needed
    printf("refine over\n");
    gmesh.nnodes = mesh.nnodes = *(nnodes.cpu_rd_ptr());
    gmesh.nelements = mesh.nelements = *(nelements.cpu_rd_ptr());
    if(debug) printf("out elements: %d\n", outwl->nitems());

    std::swap(inwl, outwl);
    outwl->reset();

    *(nbad.cpu_wr_ptr(true)) = 0;
    printf("checking triangles...\n");
    // need to check only new triangles
    //inwl->reset();
    check_triangles_orig<<<nSM, threads.x>>>(gmesh, nbad.gpu_wr_ptr(), lastnelements, *outwl, *inwl); 
    //cnbad = *(nbad.cpu_rd_ptr());
 
    cnbad = inwl->nitems();
    printf("%u bad triangles.\n", cnbad);

    if(debug) {
      debug_isbad(*inwl, mesh);
      gmesh.refresh(mesh);
    }
    if(cnbad == 0) 
      break;
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  *(nbad.cpu_wr_ptr(true)) = 0;
  check_triangles_orig<<<nSM, threads.x>>>(gmesh, nbad.gpu_wr_ptr(), 0, *outwl, *inwl); 
  cnbad = *(nbad.cpu_rd_ptr());
  printf("%d final bad triangles\n", cnbad);
  //printf("time: %f ms\n", (endtime - starttime) * 1000);
  //printf("\truntime [dmr] = %f ms\n", (endtime - starttime) * 1000);
}
