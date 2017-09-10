#!/bin/bash

#if [[ -e parsec-2.1 ]]; then
#  echo "parsec-2.1 directory already exists" >2
#  exit 1
#fi

#wget http://parsec.cs.princeton.edu/download/2.1/parsec-2.1-core.tar.gz
tar xzf parsec-2.1-core.tar.gz
find parsec-2.1 -mindepth 2 \
  ! -path 'parsec-2.1/pkgs/apps/freqmine*' \
  ! -path 'parsec-2.1/pkgs/apps/blackscholes*' \
  ! -path 'parsec-2.1/pkgs/apps/bodytrack*' \
  ! -path 'parsec-2.1/pkgs/libs/hooks*' -delete 
(cd parsec-2.1 && patch -p1 < ../parsec-2.1.patch)
