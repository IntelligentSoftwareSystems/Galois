#!/usr/bin/python

import ctypes
from ctypes import *

# type definition
GraphPtr = c_void_p
Node = c_void_p
KeyTy = c_char_p
ValTy = c_char_p

""" 
for passing struct back and forth between C++ and python, see 
http://stackoverflow.com/questions/20773602/returning-struct-from-c-dll-to-python 
"""
class Edge(Structure):
  _fields_ = [("base", c_void_p),
              ("end", c_void_p)]

glib = cdll.LoadLibrary("/net/faraday/workspace/ylu/Galois/release/exp/apps/python/libgalois_python.so")

# cast the result and arg types
glib.createGraph.restype = GraphPtr
glib.deleteGraph.argtypes = [GraphPtr]
glib.printGraph.argtypes = [GraphPtr]
glib.createNode.restype = Node
glib.createNode.argtypes = [GraphPtr]
glib.addNode.argtypes = [GraphPtr, Node]
glib.addNodeAttr.argtypes = [GraphPtr, Node, KeyTy, ValTy]
glib.removeNodeAttr.argtypes = [GraphPtr, Node, KeyTy]
glib.addEdge.restype = Edge
glib.addEdge.argtypes = [GraphPtr, Node, Node]
glib.addEdgeAttr.argtypes = [GraphPtr, Edge, KeyTy, ValTy]
glib.removeEdgeAttr.argtypes = [GraphPtr, Edge, KeyTy]
glib.analyzeBFS.argtypes = [GraphPtr, c_int]

class GaloisGraph(object):
  """Interface to a Galois graph"""

  def __init__(self):
    self.graph = glib.createGraph()
    self.nodeMap = {}
    self.edgeMap = {}

  def __del__(self):
    glib.deleteGraph(self.graph)

  def printGraph(self):
    print "=====", "GaloisGraph", "====="
    glib.printGraph(self.graph)

  def addNode(self, nid):
    if self.nodeMap.has_key(nid):
      return
    n = glib.createNode(self.graph)
    glib.addNode(self.graph, n)
    self.nodeMap[nid] = n

  def addNodeAttr(self, nid, key, val):
    glib.addNodeAttr(self.graph, self.nodeMap[nid], key, val)

  def removeNodeAttr(self, nid, key):
    glib.removeNodeAttr(self.graph, self.nodeMap[nid], key)

  def addEdge(self, eid, srcid, dstid):
    src = self.nodeMap[srcid]
    dst = self.nodeMap[dstid]
    self.edgeMap[eid] = glib.addEdge(self.graph, src, dst)

  def addEdgeAttr(self, eid, key, val):
    glib.addEdgeAttr(self.graph, self.edgeMap[eid], key, val)

  def removeEdgeAttr(self, eid, key):
    glib.removeEdgeAttr(self.graph, self.edgeMap[eid], key)

  def analyzeBFS(self, numThreads):
#     pass
    print "=====", "analyzeBFS", "====="
    print len(self.nodeMap), "nodes,", len(self.edgeMap), "edges"
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

  g.addEdge("n1n2e1", "n1", "n2")
  g.addEdgeAttr("n1n2e1", "weight", "3.0")
  g.printGraph()
  print "====="

  g.addEdge("n1n2e2", "n1", "n2")
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

