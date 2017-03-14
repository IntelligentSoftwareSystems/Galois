#!/usr/bin/python

import ctypes
from ctypes import *
import pprint

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

class NodePair(Structure):
  _fields_ = [("nQ", Node),
              ("nD", Node)]

class NodeDouble(Structure):
  _fields_ = [("n", Node),
              ("v", c_double)]

class NodeList(Structure):
  _fields_ = [("num", c_int),
              ("nodes", POINTER(Node))]

#glib = cdll.LoadLibrary("/net/faraday/workspace/ylu/Galois/debug/exp/apps/python/libgalois_python.so")
glib = cdll.LoadLibrary("/net/faraday/workspace/ylu/Galois/release/exp/apps/python/libgalois_python.so")
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
glib.searchSubgraphUllmann.restype = POINTER(NodePair)
glib.searchSubgraphUllmann.argtypes = [GraphPtr, GraphPtr, c_int]
glib.searchSubgraphVF2.restype = POINTER(NodePair)
glib.searchSubgraphVF2.argtypes = [GraphPtr, GraphPtr, c_int]
glib.deleteGraphMatches.argtypes = [POINTER(NodePair)]
glib.analyzePagerank.restype = POINTER(NodeDouble)
glib.analyzePagerank.argtypes = [GraphPtr, c_int, c_double, KeyTy]
glib.deleteNodeDoubles.argtypes = [POINTER(NodeDouble)]
glib.filterNode.argtypes = [GraphPtr, KeyTy, ValTy]
glib.filterNode.restype = NodeList
glib.deleteNodeList.argtypes = [NodeList]
glib.createNodeList.argtypes = [c_int]
glib.createNodeList.restype = NodeList
glib.findReachableFrom.argtypes = [GraphPtr, NodeList, c_int]
glib.findReachableFrom.restype = NodeList
glib.findReachableTo.argtypes = [GraphPtr, NodeList, c_int]
glib.findReachableTo.restype = NodeList
glib.findReachableBetween.argtypes = [GraphPtr, NodeList, NodeList, c_int]
glib.findReachableBetween.restype = NodeList

class GaloisGraph(object):
  """Interface to a Galois graph"""

  def __init__(self, name=""):
    self.name = name
    self.graph = glib.createGraph()
    self.nodeMap = {}
    self.edgeMap = {}

  def __del__(self):
    glib.deleteGraph(self.graph)

  def printGraph(self):
    print "=====", "GaloisGraph", self.name, "====="
    glib.printGraph(self.graph)

  def addNode(self, nid):
    if nid not in self.nodeMap:
      n = glib.createNode(self.graph)
      glib.addNode(self.graph, n)
      self.nodeMap[nid] = n
    return self.nodeMap[nid]

  def setNodeAttr(self, n, key, val):
    glib.setNodeAttr(self.graph, n, key, val)

  def getNodeAttr(self, n, key):
    return glib.getNodeAttr(self.graph, n, key)

  def removeNodeAttr(self, n, key):
    glib.removeNodeAttr(self.graph, n, key)

  def addEdge(self, eid, srcid, dstid):
    src = self.nodeMap[srcid]
    dst = self.nodeMap[dstid]
    e = glib.addEdge(self.graph, src, dst)
    self.edgeMap[eid] = e
    return e

  def setEdgeAttr(self, e, key, val):
    glib.setEdgeAttr(self.graph, e, key, val)

  def getEdgeAttr(self, e, key):
    return glib.getEdgeAttr(self.graph, e, key)

  def removeEdgeAttr(self, e, key):
    glib.removeEdgeAttr(self.graph, e, key)

  def showStatistics(self):
    print self.name, ":", len(self.nodeMap), "nodes,", len(self.edgeMap), "edges"

  def analyzeBFS(self, src, attr, numThreads):
    print "=====", "analyzeBFS", "====="
    self.showStatistics()
    print "src =", src
    glib.setNumThreads(numThreads)
    glib.analyzeBFS(self.graph, src, attr)
    print attr, "of src is", self.getNodeAttr(src, attr)

  def searchSubgraph(self, gQ, numInstances, numThreads, algo):
    print "=====", "searchSubgraph", "====="
    self.showStatistics()
    gQ.showStatistics()
    glib.setNumThreads(numThreads)

    if algo == "Ullmann":
      matches = glib.searchSubgraphUllmann(self.graph, gQ.graph, numInstances)
    elif algo == "VF2":
      matches = glib.searchSubgraphVF2(self.graph, gQ.graph, numInstances)
    else:
      raise Exception("Unknown algorithm for searchSubgraph")

    result = []
    for i in range(numInstances):
      sol = {}
      gQSize = len(gQ.nodeMap)
      for j in range(gQSize):
        nQ = matches[i*gQSize+j].nQ
        nD = matches[i*gQSize+j].nD

        if nQ != None and nD != None:
          sol[nQ] = nD
        elif nQ == None and nD == None:
          continue
        else:
          raise Exception("Matching error")

      if len(sol):
        result.append(sol);

    glib.deleteGraphMatches(matches)
    return result

  def analyzePagerank(self, topK, tolerance, attr, numThreads):
    print "=====", "analyzePagerank", "====="
    self.showStatistics()
    glib.setNumThreads(numThreads)
    pr = glib.analyzePagerank(self.graph, topK, tolerance, attr)

    result = []
    for i in range(topK):
      n = pr[i].n
      if n != None:
        result.append((n, pr[i].v))

    glib.deleteNodeDoubles(pr)
    return result

  def filterNode(self, key, value, numThreads):
    print "=====", "filterNode", "====="
    glib.setNumThreads(numThreads)
    l = glib.filterNode(self.graph, key, value)
    result = []
    for j in range(l.num):
      result.append(l.nodes[j])
    glib.deleteNodeList(l)
    return result

  def findReachableOutward(self, root, isBackward, hop, numThreads):
    print "=====", "findReachableOutward", "====="
    glib.setNumThreads(numThreads)
    rootL = glib.createNodeList(len(root))
    for i in range(len(root)):
      rootL.nodes[i] = root[i]
    if isBackward:
      print "backward within", hop, "steps"
      reachL = glib.findReachableFrom(self.graph, rootL, hop)
    else:
      print "forward within", hop, "steps"
      reachL = glib.findReachableTo(self.graph, rootL, hop)
    result = []
    for i in range(reachL.num):
      result.append(reachL.nodes[i])
    return result

  def findReachableBetween(self, srcList, dstList, hop, numThreads):
    print "=====", "findReachableBetween", "====="
    glib.setNumThreads(numThreads)
    srcL = glib.createNodeList(len(srcList))
    for i in range(len(srcList)):
      srcL.nodes[i] = srcList[i]
    dstL = glib.createNodeList(len(dstList))
    for i in range(len(dstList)):
      dstL.nodes[i] = dstList[i]
    reachL = glib.findReachableBetween(self.graph, srcL, dstL, hop)
    result = []
    for i in range(reachL.num):
      result.append(reachL.nodes[i])
    return result

def test():
  g = GaloisGraph("g")

  n0 = g.addNode("n0")
  g.setNodeAttr(n0, "color", "red")
  g.setNodeAttr(n0, "id", "node 0")

  n1 = g.addNode("n1")
  g.setNodeAttr(n1, "language", "english")
  g.setNodeAttr(n1, "garbage", "to_be_deleted")
  g.setNodeAttr(n1, "id", "node 1")

  n2 = g.addNode("n2")
  g.setNodeAttr(n2, "date", "Oct. 24, 2016")
  g.setNodeAttr(n2, "id", "node 2")
  g.printGraph()

  g.removeNodeAttr(n1, "garbage");
  g.printGraph()

  e0n0n1 = g.addEdge("e0n0n1", "n0", "n1")
  g.setEdgeAttr(e0n0n1, "weight", "3.0")
  g.setEdgeAttr(e0n0n1, "id", "edge 0: 0 -> 1")
  g.printGraph()

  e1n0n1 = g.addEdge("e1n0n1", "n0", "n1")
  g.setEdgeAttr(e1n0n1, "place", "texas")
  g.setEdgeAttr(e1n0n1, "garbage", "discard")
  g.setEdgeAttr(e1n0n1, "id", "edge 1: 0 -> 1")
  g.printGraph()

  g.removeEdgeAttr(e1n0n1, "garbage")
  g.removeEdgeAttr(e1n0n1, "galois_id")
  g.printGraph()

  e2n0n0 = g.addEdge("e2n0n0", "n0", "n0")
  g.setEdgeAttr(e2n0n0, "id", "edge 2: 0 -> 0")

  e3n1n0 = g.addEdge("e3n1n0", "n1", "n0")
  g.setEdgeAttr(e3n1n0, "id", "edge 3: 1 -> 0")
  g.printGraph()

  e4n1n2 = g.addEdge("e4n1n2", "n1", "n2")
  g.setEdgeAttr(e4n1n2, "id", "edge 4: 1 -> 2")

  e5n2n0 = g.addEdge("e5n2n0", "n2", "n0")
  g.setEdgeAttr(e5n2n0, "id", "edge 5: 2 -> 0")
  g.printGraph()

  g.analyzeBFS(n0, "dist", 1)
  g.printGraph()

  g2 = GaloisGraph("g2")
  g2.addNode("g2n0")
  g2.addNode("g2n1")
  g2.addNode("g2n2")
  g2.addEdge("e0g2n0g2n1", "g2n0", "g2n1")
  g2.addEdge("e1g2n1g2n1", "g2n1", "g2n2")
  g2.addEdge("e2g2n2g2n0", "g2n2", "g2n0")
  g2.printGraph()
  pprint.pprint(g.searchSubgraph(g2, 10, 3, "Ullmann"))
  pprint.pprint(g.searchSubgraph(g2, 10, 3, "VF2"))

  del g2

  pprint.pprint(g.analyzePagerank(10, 0.01, "pagerank", 2))

  srcList = g.filterNode("color", "red", 3)
  print "srcList:"
  pprint.pprint(srcList)
  dstList = g.filterNode("id", "node 2", 2)
  print "dstList:"
  pprint.pprint(dstList)
  fromList = g.findReachableOutward(dstList, True, 1, 2)
  print "fromList:"
  pprint.pprint(fromList)
  toList = g.findReachableOutward(srcList, False, 1, 2)
  print "toList:"
  pprint.pprint(toList)
  betweenList = g.findReachableBetween(srcList, dstList, 2, 2)
  print "betweenList:"
  pprint.pprint(betweenList)

  del g

if __name__ == "__main__":
  test()

