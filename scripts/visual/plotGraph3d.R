#!/usr/bin/Rscript

library (rgl)
library (compositions)

args = commandArgs (trailingOnly=T);

nodes = read.csv (args[1], header=T);
edges = read.csv (args[2], header=T);

nodes = nodes[rev (rownames (nodes)), ]; # order by nodeId
rownames (nodes) = 1:nrow(nodes);

edges = edges[rev (rownames (edges)), ]; # reverse like wise
rownames (edges) = 1:nrow(edges);

# scale time stamps
nodes$timeStamp = 500 * nodes$timeStamp;

localMin = nodes[nodes$inDeg == 0, ]
localMax = nodes[nodes$outDeg == 0, ]

srcVec = edges$srcId + 1
dstVec = edges$dstId + 1


open3d ();

# segX = nodes$centerX[rbind (srcVec, dstVec)]
# segY = nodes$centerY[rbind (srcVec, dstVec)]
# segZ = nodes$timeStamp[rbind (srcVec, dstVec)]
# segments3d (segX, segY, segZ);

arrowStart = cbind (nodes$centerX[srcVec], nodes$centerY[srcVec], nodes$timeStamp[srcVec]);
arrowEnd = cbind (nodes$centerX[dstVec], nodes$centerY[dstVec], nodes$timeStamp[dstVec]);
arrows3D (arrowStart, arrowEnd, angle=10, length=0.2, size=10, lwd=10);


radius=0.15

spheres3d (nodes$centerX, nodes$centerY, nodes$timeStamp, radius=radius, color=c("blue"));
spheres3d (localMin$centerX, localMin$centerY, localMin$timeStamp, radius=1.1*radius, color=c("red"));
spheres3d (localMax$centerX, localMax$centerY, localMax$timeStamp, radius=1.1*radius, color=c("green"));

# rgl.snapshot ("adjgraph3d.png")

