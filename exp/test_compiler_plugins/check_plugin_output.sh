#!/bin/bash

#eval "cp ./before_versions/$1 ./$1"

#source "/workspace/ggill/Dist_latest/build_dist_hetero/release_new_clang/exp/test_compiler_plugins/plugin_test_scripts/cc_pull_test.sh"

ls -a

eval "cat ./$1 > /dev/null"
eval "cat ./$1"

if cmp -s $1 ./after_versions/$1; then
  echo "Success: No difference found.";
else
  echo "Failed: Difference found.";
fi

