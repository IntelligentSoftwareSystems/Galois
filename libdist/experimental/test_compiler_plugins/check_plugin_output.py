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

