#!/usr/bin/python
import re
import sys
import fileinput
import getopt
import textwrap

""" remove license from the files.
    returns: text with license removed
"""
def commentRemover(text, filename):
  new_license_text = """/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
 */\n\n"""

  def replacer(match):
    s = match.group(0)
    if s.startswith('/'):
      return new_license_text # note: a space and not an empty string
    else:
      return s
  pattern = re.compile(
      #r'/\*.*?\*/',
      #r'/\*.*License.*?\*/\s+',
      r'/\*.*License.*?.*The University of Texas at Austin.*?\*/\s+',
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
