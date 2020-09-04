SPRoute
================================================================================


DESCRIPTION 
--------------------------------------------------------------------------------

This program performs global routing on a circuit. Please find our ICCAD 2019 paper "He, Jiayuan, et al. "SPRoute: A Scalable Parallel Negotiation-based Global Router." 2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD). IEEE, 2019." for details.

SPRoute is based on FastRoute 4.1 and consists of four stages: tree decomposition, pattern routing, maze routing and layer assignment. SRoute parallelizes the most time-consuming maze routing stage in a novel hybrid parallel scheme which combines net-level parallelism and fine-grain parallelism. 

INPUT
--------------------------------------------------------------------------------

Input circuit is ISPD2008 contest format. For more information please visit http://www.ispd.cc/contests/08/ispd08rc.html

Input also requires FLUTE files. Please download flute-3.1.tgz from http://home.eng.iastate.edu/~cnchu/flute.html.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/eda/cpu/sproute; make -j sproute-cpu`


RUN
--------------------------------------------------------------------------------

The following are a few example command lines.

-`$ ./sproute-cpu -ISPD2008Graph <path-to-circuit> --flute <path-to-flute-directory> -t 40`



PERFORMANCE  
--------------------------------------------------------------------------------
Please find more details in the SPRoute paper.

On a 28-core machine, SPRoute achieves an average speedup of 11X on overflow-free cases and 3.1X on hard-to-route cases in ISPD2008 benchmarks. 


