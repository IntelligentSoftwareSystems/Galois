The eclipse project has been configured to use this directory as the build directory.

Pressing Ctrl+B will run make inside this directory.

To initialize this directory you need to run cmake:
make -C ${ProjDirPath}/build VERBOSE=1

cd to the "build" directory and run "cmake .."

user@host:~/workspace/galoiscpp$ cd build
user@host:~/workspace/galoiscpp/build$ cmake ..

Contact reza@cs if you have any questions.