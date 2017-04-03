#!/usr/bin/python

from ctypes import *
import pprint

__author__ = "Yi-Shan Lu"
__email__ = "yishanlu@utexas.edu"

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
    _fields_ = [("src", Node),
                ("dst", Node)]

class EdgeList(Structure):
    _fields_ = [("num", c_int),
                ("edges", POINTER(Edge))]

class NodePair(Structure):
    _fields_ = [("nQ", Node),
                ("nD", Node)]

class NodeDouble(Structure):
    _fields_ = [("n", Node),
                ("v", c_double)]

class NodeList(Structure):
    _fields_ = [("num", c_int),
                ("nodes", POINTER(Node))]

class AttrList(Structure):
    _fields_ = [("num", c_int),
                ("key", POINTER(KeyTy)),
                ("value", POINTER(ValTy))]

#glib = cdll.LoadLibrary("/opt/galois/libgalois_python.so")
glib = cdll.LoadLibrary("/net/faraday/workspace/ylu/Galois/release/exp/apps/python/libgalois_python.so")

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
glib.getNodeAllAttr.restype = AttrList
glib.getNodeAllAttr.argtypes = [GraphPtr, Node]
glib.getAllNodes.restype = NodeList
glib.getAllNodes.argtypes = [GraphPtr]
glib.deleteAttrList.argtypes = [AttrList]
glib.removeNodeAttr.argtypes = [GraphPtr, Node, KeyTy]
glib.addEdge.restype = Edge
glib.addEdge.argtypes = [GraphPtr, Node, Node]
glib.setEdgeAttr.argtypes = [GraphPtr, Edge, KeyTy, ValTy]
glib.getEdgeAttr.restype = ValTy
glib.getEdgeAttr.argtypes = [GraphPtr, Edge, KeyTy]
glib.getEdgeAllAttr.restype = AttrList
glib.getEdgeAllAttr.argtypes = [GraphPtr, Edge]
glib.getAllEdges.restype = EdgeList
glib.getAllEdges.argtypes = [GraphPtr]
glib.removeEdgeAttr.argtypes = [GraphPtr, Edge, KeyTy]
glib.setNumThreads.argtypes = [c_int]
glib.getNumNodes.restype = c_int
glib.getNumNodes.argtypes = [GraphPtr]
glib.getNumEdges.restype = c_int
glib.getNumEdges.argtypes = [GraphPtr]
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
glib.coarsen.argtypes = [GraphPtr, GraphPtr, KeyTy]

class GaloisGraph(object):
    """Interface to a Galois graph"""

    def __init__(self, name=""):
        self.name = name
        self.graph = glib.createGraph()
        self.nodeMap = {}
        self.edgeMap = {}
        self.invNodeMap = {}
        self.invEdgeMap = {}

    def __del__(self):
        glib.deleteGraph(self.graph)
        self.nodeMap.clear()
        self.edgeMap.clear()
        self.invNodeMap.clear()
        self.invEdgeMap.clear()

    def printGraph(self):
        print "=====", "GaloisGraph", self.name, "====="
        glib.printGraph(self.graph)

    def setNodeIndex(self, n, nid):
        self.nodeMap[nid] = n
        self.invNodeMap[n] = nid

    def getNodeIndex(self, n):
        return self.invNodeMap[n]

    def addNode(self, nid):
        if nid not in self.nodeMap:
            n = glib.createNode(self.graph)
            glib.addNode(self.graph, n)
            self.setNodeIndex(n, nid)
        return self.nodeMap[nid]

    def getNode(self, nid):
        return self.nodeMap[nid]

    def getAllNodes(self):
        l = glib.getAllNodes(self.graph)
        result = []
        for j in range(l.num):
            result.append(l.nodes[j])
        glib.deleteNodeList(l)
        return result

    def setNodeAttr(self, n, key, val):
        glib.setNodeAttr(self.graph, n, key, val)

    def getNodeAttr(self, n, key):
        return glib.getNodeAttr(self.graph, n, key)

    def removeNodeAttr(self, n, key):
        glib.removeNodeAttr(self.graph, n, key)

    def getNodeAllAttr(self, n):
        attr = glib.getNodeAllAttr(self.graph, n)
        sol = {}
        for i in range(attr.num):
            sol[attr.key[i]] = attr.value[i]
        glib.deleteAttrList(attr)
        return sol

    def setEdgeIndex(self, e, eid):
        self.edgeMap[eid] = e
        self.invEdgeMap[(e.src, e.dst)] = eid

    def getEdgeIndex(self, e):
        return self.invEdgeMap[(e.src, e.dst)]

    def addEdge(self, eid, srcid, dstid):
        src = self.nodeMap[srcid]
        dst = self.nodeMap[dstid]
        e = glib.addEdge(self.graph, src, dst)
        self.setEdgeIndex(e, eid)
        return e

    def getEdge(self, eid):
        return self.edgeMap[eid]

    def getAllEdges(self):
        l = glib.getAllEdges(self.graph)
        result = []
        for j in range(l.num):
            # multi-field struct needs to be constructed
            result.append(Edge(l.edges[j].src, l.edges[j].dst))
        glib.deleteEdgeList(l)
        return result

    def setEdgeAttr(self, e, key, val):
        glib.setEdgeAttr(self.graph, e, key, val)

    def getEdgeAttr(self, e, key):
        return glib.getEdgeAttr(self.graph, e, key)

    def removeEdgeAttr(self, e, key):
        glib.removeEdgeAttr(self.graph, e, key)

    def getEdgeAllAttr(self, e):
        attr = glib.getEdgeAllAttr(self.graph, e)
        sol = {}
        for i in range(attr.num):
            sol[attr.key[i]] = attr.value[i]
        glib.deleteAttrList(attr)
        return sol

    def getNumNodes(self):
        return glib.getNumNodes(self.graph)

    def getNumEdges(self):
        return glib.getNumEdges(self.graph)

    def showStatistics(self):
        print self.name, ":", self.getNumNodes(), "nodes,", self.getNumEdges(), "edges"

    def analyzeBFS(self, src, attr, numThreads):
        print "=====", "analyzeBFS", "====="
        self.showStatistics()
        print "src =", src
        glib.setNumThreads(numThreads)
        glib.analyzeBFS(self.graph, src, attr)

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

        print "by", algo, "algorithm"
        result = []
        for i in range(numInstances):
            sol = {}
            gQSize = len(gQ.nodeMap)
            for j in range(gQSize):
                nQ = matches[i * gQSize + j].nQ
                nD = matches[i * gQSize + j].nD

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
        print "in between within", hop, "hops"
        reachL = glib.findReachableBetween(self.graph, srcL, dstL, hop)
        result = []
        for i in range(reachL.num):
            result.append(reachL.nodes[i])
        return result

    def coarsen(self, key, numThreads):
        print "=====", "coarsen", "====="
        print "by", key
        glib.setNumThreads(numThreads)
        cgName = self.name + " coarsened by " + key
        cg = GaloisGraph(cgName)
        glib.coarsen(self.graph, cg.graph, key)
        return cg

def testGraphConstruction():
    g = GaloisGraph("g")

    n0 = g.addNode("n0")
    g.setNodeAttr(n0, "color", "red")
    g.setNodeAttr(n0, "id", "node 0")
    assert(g.getNodeAttr(n0, "color") == "red")
    assert(g.getNodeAttr(n0, "id") == "node 0")

    n1 = g.addNode("n1")
    g.setNodeAttr(n1, "language", "english")
    g.setNodeAttr(n1, "garbage", "to_be_deleted")
    g.setNodeAttr(n1, "id", "node 1")
    assert(g.getNodeAttr(n1, "language") == "english")
    assert(g.getNodeAttr(n1, "garbage") == "to_be_deleted")
    assert(g.getNodeAttr(n1, "id") == "node 1")

    n2 = g.addNode("n2")
    g.setNodeAttr(n2, "date", "Oct. 24, 2016")
    g.setNodeAttr(n2, "id", "node 2")
    assert(g.getNodeAttr(n2, "date") == "Oct. 24, 2016")
    assert(g.getNodeAttr(n2, "id") == "node 2")

    g.removeNodeAttr(n1, "garbage");
    attr = g.getNodeAllAttr(n1)
    assert(2 == len(attr))
    assert("garbage" not in attr)
    assert(attr["language"] == g.getNodeAttr(n1, "language"))
    assert(attr["id"] == g.getNodeAttr(n1, "id"))

    e0n0n1 = g.addEdge("e0n0n1", "n0", "n1")
    assert(g.getNodeIndex(g.getEdge("e0n0n1").src) == "n0")
    assert(g.getNodeIndex(g.getEdge("e0n0n1").dst) == "n1")
    g.setEdgeAttr(e0n0n1, "weight", "3.0")
    g.setEdgeAttr(e0n0n1, "id", "edge 0: 0 -> 1")
    assert(g.getEdgeAttr(e0n0n1, "weight") == "3.0")
    assert(g.getEdgeAttr(e0n0n1, "id") == "edge 0: 0 -> 1")

    e1n0n1 = g.addEdge("e1n0n1", "n0", "n1")
    g.setEdgeAttr(e1n0n1, "place", "texas")
    g.setEdgeAttr(e1n0n1, "garbage", "discard")
    g.setEdgeAttr(e1n0n1, "id", "edge 1: 0 -> 1")
    assert(g.getEdgeAttr(e1n0n1, "place") == "texas")
    assert(g.getEdgeAttr(e1n0n1, "garbage") == "discard")
    assert(g.getEdgeAttr(e0n0n1, "id") == "edge 1: 0 -> 1") # overwritten

    g.removeEdgeAttr(e1n0n1, "garbage")
    g.removeEdgeAttr(e1n0n1, "galois_id")
    attr = g.getEdgeAllAttr(e1n0n1)
    assert(3 == len(attr))
    assert("garbage" not in attr) # query for removed key
    assert("galois_id" not in attr) # query for non-existed key
    assert(attr["place"] == g.getEdgeAttr(e1n0n1, "place"))
    assert(attr["weight"] == g.getEdgeAttr(e1n0n1, "weight"))
    assert(attr["id"] == g.getEdgeAttr(e1n0n1, "id"))

    e2n0n0 = g.addEdge("e2n0n0", "n0", "n0")
    g.setEdgeAttr(e2n0n0, "id", "edge 2: 0 -> 0")
    assert(g.getEdgeAttr(e2n0n0, "id") == "edge 2: 0 -> 0") # self loop

    e3n1n0 = g.addEdge("e3n1n0", "n1", "n0")
    g.setEdgeAttr(e3n1n0, "id", "edge 3: 1 -> 0")
    assert(g.getEdgeAttr(e3n1n0, "id") == "edge 3: 1 -> 0") # 2-node loop

    e4n1n2 = g.addEdge("e4n1n2", "n1", "n2")
    g.setEdgeAttr(e4n1n2, "id", "edge 4: 1 -> 2")

    e5n2n0 = g.addEdge("e5n2n0", "n2", "n0")
    g.setEdgeAttr(e5n2n0, "id", "edge 5: 2 -> 0")
    assert(g.getEdgeAttr(e4n1n2, "id") == "edge 4: 1 -> 2") # 3-node loop
    assert(g.getEdgeAttr(e5n2n0, "id") == "edge 5: 2 -> 0")

    return g

def testSubgraphIsomorphism():
    g = testGraphConstruction()

    g2 = GaloisGraph("g2")
    g2.addNode("g2n0")
    g2.addNode("g2n1")
    g2.addNode("g2n2")
    g2.addEdge("e0g2n0g2n1", "g2n0", "g2n1")
    g2.addEdge("e1g2n1g2n1", "g2n1", "g2n2")
    g2.addEdge("e2g2n2g2n0", "g2n2", "g2n0")

    # solution: mapping from query nodes to data nodes
    sol_map = [
        {"g2n0": "n0", "g2n1": "n1", "g2n2": "n2"},
        {"g2n0": "n1", "g2n1": "n2", "g2n2": "n0"},
        {"g2n0": "n2", "g2n1": "n0", "g2n2": "n1"}
    ]

    res_ull = g.searchSubgraph(g2, 10, 3, "Ullmann")
    assert(len(res_ull) == 3)
    assert(res_ull[0] != res_ull[1] and res_ull[0] != res_ull[2] and res_ull[1] != res_ull[2])
    for m in res_ull:
        # translate galois handle back to user id
        res_map = { g2.getNodeIndex(k): g.getNodeIndex(v) for k, v in m.iteritems() }
        assert(res_map == sol_map[0] or res_map == sol_map[1] or res_map == sol_map[2])

    res_vf2 = g.searchSubgraph(g2, 10, 3, "VF2")
    assert(len(res_vf2) == 3)
    assert(res_vf2[0] != res_vf2[1] and res_vf2[0] != res_vf2[2] and res_vf2[1] != res_vf2[2])
    for m in res_vf2:
        res_map = { g2.getNodeIndex(k): g.getNodeIndex(v) for k, v in m.iteritems() }
        assert(res_map == sol_map[0] or res_map == sol_map[1] or res_map == sol_map[2]) 

def testReachability():
    g = testGraphConstruction()

    srcList = g.filterNode("color", "red", 3)
    assert(len(srcList) == 1)
    assert(g.getNodeIndex(srcList[0]) == "n0")

    dstList = g.filterNode("id", "node 2", 2)
    assert(len(dstList) == 1)
    assert(g.getNodeIndex(dstList[0]) == "n2")

    fromList = g.findReachableOutward(dstList, True, 1, 2)
    assert(len(fromList) == 2)
    # map back to user id and get rid of ordering
    assert(set(g.getNodeIndex(v) for v in fromList) == set(["n1", "n2"]))

    toList = g.findReachableOutward(srcList, False, 1, 2)
    assert(len(toList) == 2)
    assert(set(g.getNodeIndex(v) for v in toList) == set(["n0", "n1"]))

    betweenList = g.findReachableBetween(srcList, dstList, 2, 2)
    assert(len(betweenList) == 3)
    assert(set(g.getNodeIndex(v) for v in betweenList) == set(["n0", "n1", "n2"]))

def testCoarsening():
    fg = GaloisGraph("fg")
    for i in range(9):
        n = fg.addNode(i)
        fg.setNodeAttr(n, "color", str(i % 3))
    fg.addNode(9)
    fg.addEdge(0, 0, 5)
    fg.addEdge(1, 1, 6)
    fg.addEdge(2, 2, 7)
    fg.addEdge(3, 3, 9)
    fg.addEdge(4, 4, 7)
    fg.addEdge(5, 5, 2)
    fg.addEdge(6, 5, 6)
    fg.addEdge(7, 8, 7)
    fg.addEdge(8, 9, 4)

    cg = fg.coarsen("color", 3)
    cg_nodes = cg.getAllNodes()
    for n in cg_nodes:
        cg.setNodeIndex(n, cg.getNodeAttr(n, "color"))
    # all nodes have different colors
    assert(set(cg.getNodeAttr(n, "color") for n in cg_nodes) == set(["0", "1", "2"]))

    cg_edges = cg.getAllEdges()
    cg_edges_tuple = []
    for e in cg_edges:
        assert(e.src != e.dst)
        cg_edges_tuple.append((cg.getNodeIndex(e.src), cg.getNodeIndex(e.dst)))
    assert(set(cg_edges_tuple) == set([("0", "2"), ("1", "0"), ("2", "0"), ("2", "1")]))

def testBFS():
    g = testGraphConstruction()
    g.analyzeBFS(g.nodeMap["n0"], "dist", 1)
    assert(g.getNodeAttr(g.getNode("n0"), "dist") == "0")
    assert(g.getNodeAttr(g.getNode("n1"), "dist") == "1")
    assert(g.getNodeAttr(g.getNode("n2"), "dist") == "2")
	
def testPagerank():
    g = testGraphConstruction()
    pr = g.analyzePagerank(10, 0.01, "pagerank", 2)
    assert(g.getNodeIndex(pr[0][0]) == "n0")
    assert(g.getNodeIndex(pr[1][0]) == "n1")
    assert(g.getNodeIndex(pr[2][0]) == "n2")
    assert(pr[0][1] >= pr[1][1] and pr[1][1] >= pr[2][1])

def test():
    print "testGraphConstruction()..."
    g = testGraphConstruction()
    del g
    print "pass"

    print "testBFS()..."
    testBFS()
    print "pass"

    print "testPagerank()..."
    testPagerank()
    print "pass"

    print "testSubgraphIsomorphism()..."
    testSubgraphIsomorphism()
    print "pass"

    print "testReachability()..."
    testReachability()
    print "pass"

    print "testCoarsening()..."
    testCoarsening()
    print "pass"

if __name__ == "__main__":
    test()

