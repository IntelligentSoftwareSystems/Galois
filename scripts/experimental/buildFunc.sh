#!/bin/bash

SRC_ROOT="$HOME/projects/GaloisCpp"

BUILD_ROOT="/workspace/$USER/build"

mkdir -p "${BUILD_ROOT}"


cc=${cc:="gcc"}
cxx=${cxx:="g++"}
build=${build:="Debug"}
cmakeOpts=${cmakeOpts:="-DUSE_PAPI=1 -DUSE_VTUNE=1 -DENABLE_DIST_GALOIS=1"}
cleanup=${cleanup:="0"}

galoisCheckStatus() {
  local cmd="$1"
  if eval "$cmd" ; then
    echo "OK: success running ($cmd)"
  else
    echo "ERROR: ($cmd) failed"
    exit -1
  fi
}

galoisSetCompilers() {
  if [[ "xx$cc" == "xxgcc" ]] ; then
    cxx="g++";
  elif [[ "xx$cc" == "xxicc" ]] ; then
    cxx="icpc";
  elif [[ "xx$cc" == "xxclang" ]] ; then
    cxx="clang++";
  else
    cxx="not found";
  fi

  galoisCheckStatus "which $cc"
  galoisCheckStatus "which $cxx"
}


galoisRunBuild() {
  galoisSetCompilers
  local buildDir=$(mktemp -d -p ${BUILD_ROOT} "$cc-$build.XXXXXX")
  galoisCheckStatus "cd $buildDir"
  galoisCheckStatus "CC=$cc CXX=$cxx cmake -DCMAKE_BUILD_TYPE=$build $cmakeOpts ${SRC_ROOT}"
  galoisCheckStatus "make -j"
  if [[ "xx$cleanup" == "xx1" ]] && [[ "xx$buildDir" != "xx" ]] ;  then
    galoisCheckStatus "rm -rf $buildDir"
  fi
}

galoisBuildMultiCompiler() {
  for c in "gcc" "clang" "icc"; do
    build="Debug"
    cc="$c"
    galoisRunBuild;

    # build="Release"
    # galoisRunBuild;
  done
}

galoisBuildMultiVer() {
  for i in 5 6 7 ; do 
    galoisCheckStatus "module load atc/1.$i"
    galoisBuildMultiCompiler
  done
}

galoisBuildGccDebug() {
  cc="gcc"
  build="Debug"
  galoisRunBuild
}

galoisBuildGccRelease() {
  cc="gcc"
  build="Release"
  galoisRunBuild
}

galoisBuildIccDebug() {
  cc="icc"
  build="Debug"
  galoisRunBuild
}

galoisBuildIccRelease() {
  cc="icc"
  build="Release"
  galoisRunBuild
}

galoisBuildClangDebug() {
  cc="clang"
  build="Debug"
  galoisRunBuild
}

galoisBuildClangRelease() {
  cc="clang"
  build="Release"
  galoisRunBuild
}
