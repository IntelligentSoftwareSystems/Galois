#!/usr/bin/env python

from galois_python import *

def constructGaloisGraphFromEdgelist(fileName, graphName):
  with open(fileName) as edgelistFile:
    g = GaloisGraph(graphName)

    for line in edgelistFile:
      edge = line.split()
      if len(edge) != 2:
        raise Exception("not a correct edgelist!")

      for node in edge:
        g.addNode(node)
      edgeIndex = str(edge[0]), "-", str(edge[1])
      g.addEdge(edgeIndex, edge[0], edge[1])

    return g

if __name__ == "__main__":
  g = constructGaloisGraphFromEdgelist("3-cycle.edgelist", "3-cycle")
  g.printGraph()

