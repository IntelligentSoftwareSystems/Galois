#!/usr/bin/python

import ctypes
from ctypes import *

glib = cdll.LoadLibrary("/net/faraday/workspace/ylu/Galois/release/exp/apps/python/libgalois_python.so")

# cast the result and arg types
glib.createGraph.restype = c_void_p
glib.deleteGraph.argtypes = [c_void_p]
glib.printGraph.argtypes = [c_void_p]
glib.createNode.restype = c_void_p
glib.createNode.argtypes = [c_void_p]
glib.addNode.argtypes = [c_void_p, c_void_p]
glib.addNodeAttr.argtypes = [c_void_p, c_void_p, c_char_p, c_char_p]
glib.removeNodeAttr.argtypes = [c_void_p, c_void_p, c_char_p]
glib.addMultiEdge.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p]
glib.addEdgeAttr.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p, c_char_p, c_char_p]
glib.removeEdgeAttr.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p, c_char_p]
glib.analyzeBFS.argtypes = [c_void_p, c_int]

class GaloisGraph(object):
  """Interface to a Galois graph"""

  def __init__(self):
#    print "__init__"
    self.graph = glib.createGraph()
#    print format(self.graph, 'x')
    self.nodeMap = {}
    self.edgeMap = {}

  def __del__(self):
#    print "__del__"
    glib.deleteGraph(self.graph)

  def printGraph(self):
#    print "printGraph"
    glib.printGraph(self.graph)

  def addNode(self, uuidN):
#    print "addNode"
    n = glib.createNode(self.graph)
#    print format(n, 'x')
    glib.addNode(self.graph, n)
    self.nodeMap[uuidN] = n

  def addNodeAttr(self, uuidN, key, val):
#    print "addNodeAttr"
    glib.addNodeAttr(self.graph, self.nodeMap[uuidN], key, val)

  def removeNodeAttr(self, uuidN, key):
    glib.removeNodeAttr(self.graph, self.nodeMap[uuidN], key)

  def addMultiEdge(self, uuidE, uuidSrc, uuidDst):
#    print "addMultiEdge"
    src = self.nodeMap[uuidSrc]
    dst = self.nodeMap[uuidDst]
#    print format(src, 'x'), format(dst, 'x')
    glib.addMultiEdge(self.graph, src, dst, uuidE)
    self.edgeMap[uuidE] = (src, dst)

  def addEdgeAttr(self, uuidE, key, val):
#    print "addEdgeAttr"
    glib.addEdgeAttr(self.graph, self.edgeMap[uuidE][0], self.edgeMap[uuidE][1], uuidE, key, val)

  def removeEdgeAttr(self, uuidE, key):
    glib.removeEdgeAttr(self.graph, self.edgeMap[uuidE][0], self.edgeMap[uuidE][1], uuidE, key)

  def analyzeBFS(self, numThreads):
    glib.analyzeBFS(self.graph, numThreads)


if __name__ == "__main__":
  g = GaloisGraph()

  g.addNode("n1")
  g.addNodeAttr("n1", "color", "red")

  g.addNode("n2")
  g.addNodeAttr("n2", "language", "english")
  g.addNodeAttr("n2", "garbage", "to_be_deleted")
  g.printGraph()
  print "====="

  g.removeNodeAttr("n2", "garbage");
  g.printGraph()
  print "====="

  g.addMultiEdge("n1n2e1", "n1", "n2")
  g.addEdgeAttr("n1n2e1", "weight", "3.0")

  g.addMultiEdge("n1n2e2", "n1", "n2")
  g.addEdgeAttr("n1n2e2", "place", "texas")
  g.addEdgeAttr("n1n2e2", "garbage", "discard")
  g.printGraph()
  print "====="

  g.removeEdgeAttr("n1n2e2", "garbage")
  g.removeEdgeAttr("n1n2e2", "galois_id")
  g.printGraph()
  print "====="

  g.analyzeBFS(1)
  g.printGraph()
  print "====="

  del g

