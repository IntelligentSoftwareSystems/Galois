rm -rf build
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_CXX_FLAGS="-I/usr/include -L/usr/lib64 -fPIC" -DCMAKE_INSTALL_PREFIX=$PREFIX -DBOOST_ROOT=$BUILD_PREFIX -DCMAKE_SKIP_INSTALL_ALL_DEPENDENCY=ON ..
make galois_shmem graph-convert graph-convert-huge -j
make install
