#!/bin/bash

eval "cp ./before_versions/$1 ./$1"

source "/workspace/ggill/Dist_latest/build_dist_hetero/release_new_clang/exp/test_compiler_plugins/plugin_test_scripts/cc_pull_test.sh"

source "/workspace/ggill/Dist_latest/dist_hetero_new/exp/test_compiler_plugins/check_plugin_output.sh" $1
