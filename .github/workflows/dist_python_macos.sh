#!/bin/bash
#
# This script cross compiles multiple MacOS python wheel packages by installing
# multiple python toolchains on a host builder.
set -e -x

SOURCE_DIR="${SOURCE_DIR:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/build}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/python-installer}"

BASE="$(cd $(dirname $0); pwd)"
PYTHON_BASE=/Library/Frameworks/Python.framework/Versions
PYTHON_VERSIONS="3.8.2 3.7.7 3.6.8"
LEAST_PYBIN="${PYTHON_BASE}/3.6/bin"

for ver in ${PYTHON_VERSIONS}; do
  name=python-${ver}-macosx10.9.pkg
  url=https://www.python.org/ftp/python/${ver}/${name}

  if [[ ! -f "${CACHE_DIR}/${name}" ]]; then
    mkdir -p "${CACHE_DIR}"
    (cd "${CACHE_DIR}" && curl -fL -O "$url")
  fi

  sudo installer -pkg "${CACHE_DIR}/${name}" -target /
done

"${BASE}/setup_conan.sh"

conan install -if /tmp/conan --build=missing "${SOURCE_DIR}/config"

for ver in ${PYTHON_VERSIONS}; do
  short_ver=${ver:0:3}
  pybin="${PYTHON_BASE}/${short_ver}/bin"
  if [[ -f "${SOURCE_DIR}/dev-requirements.txt" ]]; then
    "${pybin}/pip3" install -r "${SOURCE_DIR}/dev-requirements.txt"
  fi

  export GALOIS_CMAKE_ARGS="\
    -DCMAKE_TOOLCHAIN_FILE=/tmp/conan/conan_paths.cmake \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DUSE_NATIVE=none \
    "

  "${pybin}/pip3" wheel -w /tmp/wheelhouse "${SOURCE_DIR}"

  unset GALOIS_CMAKE_ARGS
done

"${LEAST_PYBIN}/pip3" install delocate
for whl in /tmp/wheelhouse/*.whl; do
  "${LEAST_PYBIN}/delocate-wheel" "${whl}" -w "${OUTPUT_DIR}"
done
