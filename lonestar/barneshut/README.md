
DESCRIPTION 
===========

This program performs N-body simulation using Barnes-Hut algorithm

The simulation proceeds in rounds (specified via -steps), where in every round, it creates an Oct-Tree of 
the bodies (specified via -n) and performs force computation between all pairs of bodies while
traversing the Oct-Tree. 


INPUT
===========

Input is randomly generated (using Plummer model) upon running the program

BUILD
===========

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/barneshut; make -j`


RUN
===========

The following are a few example command lines.

-`$ ./barneshut -n 12345 -t 40`
-`$ ./barneshut -n 12345 -steps 100 -t 40`



PERFORMANCE  
===========
- CHUNK_SIZE needs to be tuned for machine and input. 
