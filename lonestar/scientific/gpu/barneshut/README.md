## DESCRIPTION

This benchmark simulates the gravitational forces acting on a galactic cluster using the Barnes-Hut n-body algorithm. 
The positions and velocities of the n galaxies are initialized according to the empirical Plummer model. 
The program calculates the motion of each galaxy through space for a number of time steps. 
The data parallelism in this algorithm arises primarily from the independent force calculations.

## BUILD

Assuming CMake is performed in the ${GALOIS\_ROOT}/build, compile the application by executing the
following command in the ${GALOIS\_ROOT}/build/lonestar/scientific/gpu/barneshut directory.

`make -j`

## RUN

Execute as: barneshut <bodies> <timesteps> <deviceid>
e.g., ./barneshut 50000 2 0
