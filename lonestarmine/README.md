Overview of Graph Pattern Mining (GPM) in Galois
================================================================================

This directory contains benchmarks that run using the Pangolin library[1].

[1] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali, 
Pangolin: An Efficient and Flexible Graph Pattern Mining System on CPU and GPU, arXiv:1911.06969

BUILD
===========

1. Run cmake at BUILD directory `cd build; cmake -DUSE_PANGOLIN=1 ../`
To enable GPU mining, do `cmake -DUSE_PANGOLIN=1 -DUSE_GPU=1 ../`

2. Run `cd <BUILD>/lonestar/experimental/fsm; make -j`

RUN
===========

The following are a few example command lines.

-`$ ./tc <path-to-graph> -t 40`
-`$ ./kcl <path-to-graph> -k=3 -t 40`
-`$ ./motif <path-to-graph> -k=3 -t 40`
-`$ ./fsm <path-to-graph> -k=3 -minsup=300 -t 40`


