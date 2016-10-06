#!/usr/bin/python

import ctypes
from ctypes import *

glib = cdll.LoadLibrary("./libgalois_python_so.so")

# cast the result and arg types
glib.createGraph.restype = c_void_p
glib.deleteGraph.argtypes = [c_void_p]
glib.createNode.restype = c_void_p
glib.createNode.argtypes = [c_void_p]
glib.addNode.argtypes = [c_void_p, c_void_p]
glib.addNodeAttr.argtypes = [c_void_p, c_void_p, c_char_p, c_char_p]
glib.removeNodeAttr.argtypes = [c_void_p, c_void_p, c_char_p]
glib.addEdge.restype = c_void_p
glib.addEdge.argtypes = [c_void_p, c_void_p, c_void_p]
glib.addMultiEdge.restype = c_void_p
glib.addMultiEdge.argtypes = [c_void_p, c_void_p, c_void_p]
glib.addEdgeAttr.argtypes = [c_void_p, c_void_p, c_char_p, c_char_p]
glib.removeEdgeAttr.argtypes = [c_void_p, c_void_p, c_char_p]
glib.analyzeBFS.argtypes = [c_void_p, c_int]

class GaloisGraph(object):
  """Interface to a Galois graph"""

  def __init__(self):
    self.graph = glib.createGraph()
    self.nodeMap = {}
    self.edgeMap = {}

  def __del__(self):
    glib.deleteGraph(self.graph.value)

  def addNode(self, uuidN):
    n = glib.createNode(self.graph.value)
    glib.addNode(self.graph.value, n.value)
    self.nodeMap[uuidN] = n

  def addNodeAttr(self, uuidN, key, val):
    glib.addNodeAttr(self.graph.value, nodeMap[uuidN].value, key, val)

  def removeNodeAttr(self, uuidN, key):
    glib.removeNodeAttr(self.graph.value, nodeMap[uuidN].value, key)

  def addEdge(self, uuidE, uuidSrc, uuidDst):
    src = nodeMap[uuidSrc]
    dst = nodeMap[uuidDst]
    edgeMap[uuidE] = glib.addEdge(self.graph.value, src.value, dst.value)

  def addMultiEdge(self, uuidE, uuidSrc, uuidDst):
    src = nodeMap[uuidSrc]
    dst = nodeMap[uuidDst]
    edgeMap[uuidE] = glib.addMultiEdge(self.graph.value, src.value, dst.value)

  def addEdgeAttr(self, uuidE, key, val):
    glib.addEdgeAttr(self.graph.value, edgeMap[uuidE].value, key, val)

  def removeEdgeAttr(self, uuidEdge, key):
    glib.removeEdgeAttr(self.graph.value, edgeMap[uuidE].value, key)

  def analyzeBFS(self, numThreads):
    glib.analyzeBFS(self.graph.value, numThreads)


if __name__ == "__main__":
  g = GaloisGraph()
  g.addNode("n1")
  g.addNodeAttr("n1", "color", "red")
  g.addNode("n2")
  g.addNodeAttr("n2", "language", "english")
  g.addMultiEdge("n1n2e1", "n1", "n2")
  g.addEdgeAttr("n1n2e1", "weight", "3.0")
  g.analyzeBFS(1)

