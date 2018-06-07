DESCRIPTION 
===========

This is a heuristic SAT solver based on Bayesian inference. We implement the 
algorithm from the the following paper:

- A. Braunstein, M. Mezard, and R. Zecchina. Survey Propagation: An Algorithm for 
Satisfiability. Random Structures and Algorithms, 27:201-226, 2005.


INPUT
===========

The implementaiton generates a random boolean formula to solve. See example command lines for details.


BUILD
===========

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/surveypropagation; make -j`


RUN
===========

The following is an example command line.

-`$ ./surveypropagation <random-seed> <num-literals> <num-clauses> <num-literals-per-clause> -t <num-threads>`


PERFORMANCE
===========

The performance depends on the followings:

- For the do_all loops named "update_biases" and "fix_variables", tune the 
compile time constant, CHUNK_SIZE, the granularity of stolen work when work 
stealing is enabled (via galois::steal()).

- For the for_each loop named "update_eta", tune the compile time constant,
CHUNK_SIZE, the granularity of work distribution used by galois::wl<WL>().

- The optimal values of the constants might depend on the architecture, so you 
might want to evaluate the performance over a range of values (say [16-4096]).
