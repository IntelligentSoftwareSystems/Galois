Points To Analysis
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Points-to analysis based on Hardekopf and Lin's points-to analysis algorithm.

Given a constraint file (format detailed below), runs a graph based points-to
analysis algorithm to determine which nodes point to which other nodes.
Both a serial and a multi-threaded version exist, and the serial version
supports online cycle detection.

Performance is achieved by using a sparse bit vector to represent both
edges and points-to information.

INPUT
--------------------------------------------------------------------------------

The input is a constraint file in the following format:

```
<num vars> <num constraints>
<constraint num> <src> <dst> <type> <offset>
<constraint num> <src> <dst> <type> <offset>
<constraint num> <src> <dst> <type> <offset>
.
.
.
<constraint num> <src> <dst> <type> <offset>
<EOF>
```

`<src>` and `<dst>` are node IDs, and `<type>` specifies the relation
between them. `<offset>` is not supported in the implementation: it must be
set to 0. If it is not, the entire constraint will be ignored.

The constraint types supported are the following:

0 = Address Of Constraint
1 = Copy Constraint
2 = Load Constraint
3 = Store Constraint

All other constraint types will be ignored.

Note that the correctness of the parallel version is relative to the serial
version, which may or may not match other implementations of points-to
analysis.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/cpu/pointstoanalysis; make -j`

RUN
--------------------------------------------------------------------------------

Run serial points-to analysis with the following command:
`./pointstoanalysis-cpu <constraint file> -serial`

Run serial points-to analysis with online cycle detection with the following 
command:
`./pointstoanalysis-cpu <constraint file> -serial -ocd`

Run serial points-to analysis that reprocesses load/store constraints after
N constraints with the following command:
`./pointstoanalysis-cpu <constraint file> -serial -lsThreshold=N`

Run the parallel version of points-to analysis with the following command:
`./pointstoanalysis-cpu <constraint file> -t=<num threads>`

Run the parallel version of points-to analysis and print the results with
the following command (the serial version also supports printAnswer):
`./pointstoanalysis-cpu <constraint file> -t=<num threads> -printAnswer`

PERFORMANCE  
--------------------------------------------------------------------------------

Online cycle detection in the serial version may or may not help depending on the
input. There are cases where it can hurt performance. The serial version also
has a threshold that determines load/store constraints are reprocessed.
Depending on your input, you may get better performance by tuning the frequency
at which these constraints are reprocessed (the idea is that it may eliminate
redundant constraints that currently exist in the worklist).
