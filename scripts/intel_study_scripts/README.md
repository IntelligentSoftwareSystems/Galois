Instructions to build Galois and reproduce IntelStudy experiments
========

Clone the repository
```Shell
git clone https://github.com/IntelligentSoftwareSystems/Galois
```

Let us assume that the SRC_DIR is the top-level Galois source dir where the Galois repository is cloned.

Building Galois
------------
```Shell
BUILD_DIR=<path-to-your-build-dir>
mkdir -p $BUILD_DIR
cmake -S $SRC_DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release
```

Galois applications are in lonestar directory. In order to build a particular application:
```Shell
make -C $BUILD_DIR/lonestar/analytics/cpu/<app-dir-name> -j
```

For IntelStudy build the following apps:
------------
BFS
```Shell
make -C $BUILD_DIR/lonestar/analytics/cpu/bfs -j
```
BC
```Shell
make -C $BUILD_DIR/lonestar/analytics/cpu/betweennesscentrality -j
```
CC
```Shell
make -C $BUILD_DIR/lonestar/analytics/cpu/connectedcomponents -j
```
PR
```Shell
make -C $BUILD_DIR/lonestar/analytics/cpu/pagerank -j
```
SSSP
```Shell
make -C $BUILD_DIR/lonestar/analytics/cpu/sssp -j
```
TC
```Shell
make -C $BUILD_DIR/lonestar/analytics/cpu/triangles -j
```


Download the inputs
------------
```Shell
mkdir -p $INPUT_DIR
bash $BUILD_DIR/scripts/intel_study_scripts/download_inputs.sh $INPUT_DIR
```


Running benchmarks using scripts:
------------

Set env variables to be used by scripts
```Shell
export GALOIS_BUILD=$BUILD_DIR
export INPUT_DIR=$INPUT_DIR
```

Run
```Shell
cd $BUILD_DIR/scripts/intel_study_scripts/
./run_bc.sh
./run_bfs.sh
./run_cc.sh
./run_pr.sh
./run_sssp.sh
./run_tc.sh
```

logs will be produced by the above mentioned scripts in the repespective folders of the benchmark, here is the example for bfs:
```Shell
cd $BUILD_DIR/lonestar/analytics/cpu/bfs/logs
```

