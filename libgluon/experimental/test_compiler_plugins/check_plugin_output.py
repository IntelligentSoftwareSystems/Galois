/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

import subprocess
import shutil
import sys, getopt
import filecmp
import shlex

def run_check(inputFile):
  subprocess.call("ls")
  src = './before_versions/' + inputFile
  dst = inputFile

  shutil.copy2(src, dst)

  path_to_script = "/workspace/ggill/Dist_latest/build_dist_hetero/release_new_clang/exp/test_compiler_plugins/plugin_test_scripts/run_plugins.sh"

  print "Path to Script : " , path_to_script

  arg_to_sript = inputFile.split('.')[0]

  script_cmd = path_to_script + ' ' + arg_to_sript

  #subprocess.call(path_to_script + ' ' + arg_to_sript, shell=True)
  subprocess.call(shlex.split(script_cmd))

  subprocess.call("ls")

  cmp_toFile = './after_versions/' + inputFile
  if filecmp.cmp(inputFile, cmp_toFile):
    print "SUCCESS: MATCHED!!"
  else:
    print "FAILED: SOMETHING's WRONG!!"





def main(argv):
  inputFile = ''

  try:
    opts, args = getopt.getopt(argv, "hi:",["ifile="])
  except getopt.GetoptError:
    print 'python check_plugin_output.py -i <input cpp file>'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print 'python check_plugin_output.py -i <input cpp file>'
      sys.exit()
    elif opt in ("-i", "--ifile"):
      inputFile = arg

  print "Input cpp file being checked : ", inputFile

  run_check(inputFile)


if __name__ == "__main__":
  main(sys.argv[1:])

