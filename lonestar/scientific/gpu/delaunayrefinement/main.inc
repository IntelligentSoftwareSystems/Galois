/* -*- mode: c++ -*- */
int main(int argc, char *argv[]) {
  ShMesh mesh;
  int maxfactor = 20;
  int mesh_nodes, mesh_elements;
  dim3 blocks(14*8), threads(TB_SIZE);
  int no_output = 0;

  if(argc == 1)
    {
      printf("Usage: %s basefile [maxfactor] [--no]\n", argv[0]);
      exit(0);
    }

  if(argc >= 3)
    {
      if(strcmp(argv[2], "--no") == 0)
	no_output = 1;
      else
	maxfactor = atoi(argv[2]);
    }

  if(argc == 4)
    {
      if(strcmp(argv[3], "--no") == 0)
	no_output = 1;
    }

  read_mesh(argv[1], mesh, maxfactor);
  mesh_nodes = mesh.nnodes; mesh_elements = mesh.ntriangles + mesh.nsegments;

  refine_mesh(mesh, blocks, threads);
  printf("%f increase in number of elements (maxfactor hint)\n", 1.0 * mesh.nelements / mesh_elements);
  printf("%f increase in number of nodes (maxfactor hint)\n", 1.0 * mesh.nnodes / mesh_nodes);

  verify_mesh(mesh);
  if(!no_output)
    write_mesh(argv[1], mesh);
  else
    printf("Not writing output.\n");

  return 0;
}
