Maximal Independent Set
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------

This benchmark computes the maximal independent set in an unweighted graph.

INPUT
--------------------------------------------------------------------------------

This application takes in Galois .gr graphs.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/gpu/independentset; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm, use the following:

-`$ ./minimum-spanningtree-gpu -o=<output-file> <input-graph>`

-`$ ./minimum-spanningtree-gpu -o outfile.txt road-USA.gr`
