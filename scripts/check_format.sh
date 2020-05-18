#!/bin/bash

CLANG_FORMAT=${CLANG_FORMAT:-clang-format}
set -e

if [ $# -eq 0 ]; then
  echo "$(basename $0) [-fix] <paths>" >&2
  exit 1
fi

FIX=
if [ "$1" == "-fix" ]; then
  FIX=1
  shift 1
fi

ROOTS="$@"
FAILED=

while read -d '' filename; do
  if [ -n "${FIX}" ]; then
    echo "fixing ${filename}"
    ${CLANG_FORMAT} -style=file -i "${filename}"
  else
    if ${CLANG_FORMAT} -style=file -output-replacements-xml "${filename}" | grep '<replacement ' > /dev/null; then
        echo "${filename} NOT OK"
        FAILED=1
    fi
  fi
done < <(find ${ROOTS} -name experimental -prune -o -name external -prune -o -name '*.cpp' -print0 -o -name '*.h' -print0)

if [ -n "${FAILED}" ]; then
  exit 1
fi
