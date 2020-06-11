Barnes-Hut N-body Simulation
================================================================================

DESCRIPTION
--------------------------------------------------------------------------------

This benchmark simulates the gravitational forces acting on a galactic cluster
using the Barnes-Hut n-body algorithm. The positions and velocities of the n
galaxies are initialized according to the empirical Plummer model. The program
calculates the motion of each galaxy through space for a number of time steps.
The data parallelism in this algorithm arises primarily from the independent
force calculations.

BUILD
--------------------------------------------------------------------------------

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/analytics/gpu/barneshut; make -j`

RUN
--------------------------------------------------------------------------------

To run default algorithm, use the following:

-`$ ./barneshut-gpu  <bodies> <timesteps> <deviceid>`

-`$ ./barneshut-gpu 50000 2 0`
