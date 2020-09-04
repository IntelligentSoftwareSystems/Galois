Delaunayrefinement
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

The LSG Delaunay Mesh Refinement uses a variant of Chew's algorithm as
implemented in the Lonestar CPU benchmark.

A great resource on Delaunary Mesh Refinement is the website
maintained by Shewchuk:

https://www.cs.cmu.edu/~quake/triangle.research.html


INPUT
--------------------------------------------------------------------------------

Test inputs (files with extensions .ele, .node, .poly) can be downloaded
from [https://www.cs.cmu.edu/~quake/triangle.html](this url)

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/scientific/gpu/delaunayrefinement; make -j`

RUN
--------------------------------------------------------------------------------

The following are a few example command lines.

- `$ ./delaunayrefinement-gpu <input-basename> <maxfactor>`
- `$ ./delaunayrefinement-gpu r1M 20`

PERFORMANCE  
--------------------------------------------------------------------------------

* In our experience, nondet schedule in  delaunayrefinement outperforms deterministic schedules, because determinism incurs a performance cost
* Performance is sensitive to CHUNK_SIZE for the worklist, whose optimal value is input and
  machine dependent
