#!/usr/bin/python
import re
import sys
import fileinput
import getopt

""" remove license from the files.
    returns: text with license removed
"""
def commentRemover(text, filename):
  def replacer(match):
    s = match.group(0)
    if s.startswith('/'):
      return "" # note: a space and not an empty string
    else:
      return s
  pattern = re.compile(
      #r'/\*.*?\*/',
      r'/\*.*License.*?\*/\s+',
      re.DOTALL | re.MULTILINE
  )
  return re.sub(pattern, replacer, text, 1)

def main(argv):
  inputfile = ''
  outputfile = ''
  try:
    opts, args = getopt.getopt(argv,"hi:",["ifile="])
  except getopt.GetoptError:
    print 'remove_galois_license.py -i <inputfile>'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print 'remove_galois_license.py -i <inputfile>'
      sys.exit()
    elif opt in ("-i", "--ifile"):
      inputfile = arg
    print 'Input file is "', inputfile

    filename = inputfile
  with open(filename, 'r+') as f:
       uncmtFile = commentRemover(f.read(), filename)
       #print uncmtFile
       f.seek(0)
       f.write(uncmtFile)
       f.truncate()
       #print uncmtFile

if __name__ == "__main__":
  main(sys.argv[1:])
