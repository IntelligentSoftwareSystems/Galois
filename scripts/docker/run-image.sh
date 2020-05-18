#!/bin/bash
#
# This script runs a development environment inside a docker container. This
# can be useful if you don't want to or have difficulty installing dependencies
# on your host machine.
#
# In order to use this script, you must first build the quick-dev image:
#
#   docker build -t quick-dev .
#
ROOT_DIR=$(cd $(dirname $0)/../..; pwd)

IMAGE=${IMAGE:-quick-dev}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/quick-dev}

if [[ -z "${DOCKER_USER}" ]]; then
  DOCKER_USER="$(id -u):$(id -g)"
fi

cat<<EOF
###############################################
The following commands will create a working build:

  mkdir build
  conan install -if build --build=missing config
  cmake -S . -B build \\
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \\
    -DCMAKE_TOOLCHAIN_FILE=build/conan_paths.cmake
  make -C build -j 4

If you need to become root in the container:

  gosu root whoami

Because your user ID does not exist in this container, you may see errors
related to missing group or user IDs. You can ignore them.
###############################################

EOF

mkdir -p "$CACHE_DIR/conan" "$CACHE_DIR/ccache"

exec docker run --rm -it \
  --user "$DOCKER_USER" \
  -v "$ROOT_DIR":/source \
  -v "$CACHE_DIR/conan":/root/.conan/data \
  -v "$CACHE_DIR/ccache":/root/.ccache \
  $IMAGE "$@"
