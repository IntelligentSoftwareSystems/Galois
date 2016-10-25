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

glib = cdll.LoadLibrary("/net/faraday/workspace/ylu/Galois/debug/exp/apps/python/libgalois_python.so")
#glib = cdll.LoadLibrary("/net/faraday/workspace/ylu/Galois/release/exp/apps/python/libgalois_python.so")
#glib = cdll.LoadLibrary("/home/lenharth/UT/build/GaloisSM/debug/exp/apps/python/libgalois_python.so")

# cast the result and arg types
glib.createGraph.restype = GraphPtr
glib.deleteGraph.argtypes = [GraphPtr]
glib.printGraph.argtypes = [GraphPtr]
glib.createNode.restype = Node
glib.createNode.argtypes = [GraphPtr]
glib.addNode.argtypes = [GraphPtr, Node]
glib.setNodeAttr.argtypes = [GraphPtr, Node, KeyTy, ValTy]
glib.getNodeAttr.restype = ValTy
glib.getNodeAttr.argtypes = [GraphPtr, Node, KeyTy]
glib.removeNodeAttr.argtypes = [GraphPtr, Node, KeyTy]
glib.addEdge.restype = Edge
glib.addEdge.argtypes = [GraphPtr, Node, Node]
glib.setEdgeAttr.argtypes = [GraphPtr, Edge, KeyTy, ValTy]
glib.getEdgeAttr.restype = ValTy
glib.getEdgeAttr.argtypes = [GraphPtr, Edge, KeyTy]
glib.removeEdgeAttr.argtypes = [GraphPtr, Edge, KeyTy]
glib.setNumThreads.argtypes = [c_int]
glib.analyzeBFS.argtypes = [GraphPtr, Node, KeyTy]

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
#      "repeated", nid, "=> Node", format(self.nodeMap[nid], 'x')
      return
    n = glib.createNode(self.graph)
    glib.addNode(self.graph, n)
    self.nodeMap[nid] = n
#    print "add", nid, "=> Node", format(n, 'x')

  def setNodeAttr(self, nid, key, val):
    glib.setNodeAttr(self.graph, self.nodeMap[nid], key, val)

  def getNodeAttr(self, nid, key):
    return glib.getNodeAttr(self.graph, self.nodeMap[nid], key)

  def removeNodeAttr(self, nid, key):
    glib.removeNodeAttr(self.graph, self.nodeMap[nid], key)

  def addEdge(self, eid, srcid, dstid):
#    print "trying to add edge from", srcid, "to", dstid
#    if self.nodeMap.has_key(srcid):
#      print "find", srcid, "=> Node", format(self.nodeMap[srcid], 'x')
#    if self.nodeMap.has_key(dstid):
#      print "find", dstid, "=> Node", format(self.nodeMap[dstid], 'x')
    src = self.nodeMap[srcid]
    dst = self.nodeMap[dstid]
    self.edgeMap[eid] = glib.addEdge(self.graph, src, dst)
#    print "add", eid, "=> Edge (", format(self.edgeMap[eid].base, 'x'), ",", format(self.edgeMap[eid].end, 'x'), ")"

  def setEdgeAttr(self, eid, key, val):
    glib.setEdgeAttr(self.graph, self.edgeMap[eid], key, val)

  def getEdgeAttr(self, eid, key):
    return glib.getEdgeAttr(self.graph, self.edgeMap[eid], key)

  def removeEdgeAttr(self, eid, key):
    glib.removeEdgeAttr(self.graph, self.edgeMap[eid], key)

  def analyzeBFS(self, srcid, attr, numThreads):
#     pass
    print "=====", "analyzeBFS", "====="
    print len(self.nodeMap), "nodes,", len(self.edgeMap), "edges"
    print "src =", srcid
    glib.setNumThreads(numThreads)
    glib.analyzeBFS(self.graph, self.nodeMap[srcid], attr)
    print attr, "of src is", self.getNodeAttr(srcid, attr)


if __name__ == "__main__":
  g = GaloisGraph()

  g.addNode("n0")
  g.setNodeAttr("n0", "color", "red")
  g.setNodeAttr("n0", "id", "node 0")

  g.addNode("n1")
  g.setNodeAttr("n1", "language", "english")
  g.setNodeAttr("n1", "garbage", "to_be_deleted")
  g.setNodeAttr("n1", "id", "node 1")

  g.addNode("n2")
  g.setNodeAttr("n2", "date", "Oct. 24, 2016")
  g.setNodeAttr("n2", "id", "node 2")
  g.printGraph()

  g.removeNodeAttr("n1", "garbage");
  g.printGraph()

  g.addEdge("e0n0n1", "n0", "n1")
  g.setEdgeAttr("e0n0n1", "weight", "3.0")
  g.setEdgeAttr("e0n0n1", "id", "edge 0: 0 -> 1")
  g.printGraph()

  g.addEdge("e1n0n1", "n0", "n1")
  g.setEdgeAttr("e1n0n1", "place", "texas")
  g.setEdgeAttr("e1n0n1", "garbage", "discard")
  g.setEdgeAttr("e1n0n1", "id", "edge 1: 0 -> 1")
  g.printGraph()

  g.removeEdgeAttr("e1n0n1", "garbage")
  g.removeEdgeAttr("e1n0n1", "galois_id")
  g.printGraph()

  g.addEdge("e2n0n0", "n0", "n0")
  g.setEdgeAttr("e2n0n0", "id", "edge 2: 0 -> 0")

  g.addEdge("e3n1n0", "n1", "n0")
  g.setEdgeAttr("e3n1n0", "id", "edge 3: 1 -> 0")
  g.printGraph()

  g.addEdge("e4n1n2", "n1", "n2")
  g.setEdgeAttr("e4n1n2", "id", "edge 4: 1 -> 2")

  g.addEdge("e5n2n0", "n2", "n0")
  g.setEdgeAttr("e5n2n0", "id", "edge 5: 2 -> 0")
  g.printGraph()

  g.analyzeBFS("n0", "dist", 1)
  g.printGraph()

  del g

