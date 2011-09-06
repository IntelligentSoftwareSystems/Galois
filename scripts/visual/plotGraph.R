#!/usr/bin/Rscript

library (rgl)

args = commandArgs (trailingOnly=T);

nodes = read.csv (args[1], header=T);
edges = read.csv (args[2], header=T);

nodes = nodes[rev (rownames (nodes)), ]; # order by nodeId
rownames (nodes) = 1:nrow(nodes);

edges = edges[rev (rownames (edges)), ]; # reverse like wise
rownames (edges) = 1:nrow(edges);

localMin = nodes[nodes$inDeg == 0, ]
localMax = nodes[nodes$outDeg == 0, ]

srcVec = edges$srcId + 1
dstVec = edges$dstId + 1

arrowCoord = cbind (nodes$centerX[srcVec], nodes$centerY[srcVec], nodes$centerX[dstVec], nodes$centerY[dstVec]);

cex=2.5;
pch=20

outfile = "adjgraph.pdf"
pdf (outfile)
plot (nodes$centerX, nodes$centerY, type="p", pch=pch, col="transparent", cex=cex, xlab="", ylab="");

arrows (arrowCoord[,1], arrowCoord[,2], arrowCoord[,3], arrowCoord[,4], length=0.1, angle=7, lwd=1);

points (nodes$centerX, nodes$centerY, pch=pch, col="blue", cex=cex);

points (localMin$centerX, localMin$centerY, pch=pch, col="red", cex=cex);

points (localMax$centerX, localMax$centerY, pch=pch, col="green", cex=cex);


dev.off ();
