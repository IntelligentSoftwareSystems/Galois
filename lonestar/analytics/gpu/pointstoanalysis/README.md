Points To Analysis
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------

Given a set of points-to constraints, the problem is to compute the points-to
information for each pointer, in a flow-insensitive context-insensitive manner.

INPUT
--------------------------------------------------------------------------------

This application takes in Galois .gr graphs representing constraints.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/gpu/pointstoanalysis; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm, use the following:

-`$ ./pta <nodes-file> <constraints-file> <hcd-table solution-file> [TRANSFER, VERIFY]
