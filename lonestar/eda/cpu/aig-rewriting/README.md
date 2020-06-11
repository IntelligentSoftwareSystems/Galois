Aig-Rewriting
================================================================================


DESCRIPTION 
--------------------------------------------------------------------------------

This program rewrites a given AIG in order to reduce the number of AIG nodes
while preseving the functional equivalence of the represented circuit. For 
details, please refer to the following paper:

Vinicius Possani, Yi-Shan Lu, Alan Mishchenko, Keshav Pingali, Renato Ribas, 
André Reis. Unlocking Fine-Grain Parallelism for AIG Rewriting. In ICCAD 2018.


INPUT
--------------------------------------------------------------------------------

The program expects an AIG graph.


BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/eda/cpu/aig-rewriting; make -j aig-rewriting-cpu`


RUN
--------------------------------------------------------------------------------

The following are a few example command lines.

-`$ ./aig-rewriting-cpu <path-AIG> -t 14`
-`$ ./aig-rewriting-cpu <path-AIG> -t 28 -v`


PERFORMANCE  
--------------------------------------------------------------------------------

- Performance is sensitive to CHUNK_SIZE for the worklist, whose optimal value is input and
  machine dependent
