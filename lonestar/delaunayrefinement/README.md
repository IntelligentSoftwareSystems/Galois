DESCRIPTION 
===========

This program refines a 2D Delaunay Mesh such that no angle in any triangles is less
than a certain value (30 deg in this implementation) 

This implementation contains both non-deterministic and deterministic parallel
schedules for refining the mesh. 

INPUT
===========

The user specifies a *basename* of 3 files read by delaunayrefinement:
1. basename.nodes contains positions of vertices/points
2. basename.ele contains info about vertices of triangles
3. basename.poly contains info about which triangles are adjacent to each other


BUILD
===========

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/delaunayrefinement; make -j`


RUN
===========

The following are a few example command lines.

- `$ ./delaunayrefinement <input-basename> -t 40`
- `$ ./delaunayrefinement <input-basename> -detPrefix -t 40` for one of the
  available deterministic schedules



TUNING PERFORMANCE  
==================

- In our experience, nondet schedule in  delaunayrefinement outperforms deterministic schedules, because determinism incurs a performance cost
- Performance is sensitive to CHUNK_SIZE for the worklist, whose optimal value is input and
  machine dependent
