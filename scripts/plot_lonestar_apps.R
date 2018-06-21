library(ggplot2)
library(gtable)
library(grid)
library(gridExtra)
library(plyr)
library(reshape2)
#library(dplyr)

showPlot <- function(name, p, ...) {
  if (interactive()) {
    return(invisible(NULL))
  }
  outfile <- paste("figs", "/", name, ".pdf", sep="")
  ggsave(p + theme(plot.margin=unit(c(0, 0.25, 0, 0), "lines")), file=outfile, ...)
}

showImage <- function(name, p, ...) {
  if (interactive()) {
    return(invisible(NULL))
  }
  outfile <- paste("figs", "/", name, ".png", sep="")
  ggsave(p + theme(plot.margin=unit(c(0, 0.25, 0, 0), "lines")), file=outfile, width=8, height=4, ...)
}

showPlotGrid <- function(name, p, ...) {
    if (interactive()) {
          return(invisible(NULL))
  }
  outfile <- paste("figs", "/", name, ".pdf", sep="")
    ggsave(p , file=outfile, ...)
}

geomean <- function(label, x) {
  gm <- exp(sum(log(x[x > 0]), na.rm=TRUE) / length(x))
  print(c(label, gm))
}

round_df <- function(df, digits) {
  nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))

  df[,nums] <- round(df[,nums], digits = digits)

  (df)
}

grid_arrange_shared_legend <- function(..., ncol = length(list(...)), nrow = 1, position = c("bottom", "right"), name = "comm-breakdown") {

  plots <- list(...)
  position <- match.arg(position)
  name <- match.arg(name)
  g <- ggplotGrob(plots[[1]] +labs(fill="") + guides(fill = guide_legend(nrow = 2)) + theme(legend.position = position))$grobs
  legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
  lheight <- sum(legend$height)
  lwidth <- sum(legend$width)
  gl <- lapply(plots, function(x) x + theme(legend.position="none"))
  gl <- c(gl, ncol = ncol, nrow = nrow)

  combined <- switch(position,
                     "bottom" = arrangeGrob(do.call(arrangeGrob, gl),
                                            legend,
                                            ncol = 1,
                                            heights = unit.c(unit(1, "npc") - lheight, lheight)),
                     "right" = arrangeGrob(do.call(arrangeGrob, gl),
                                           legend,
                                           ncol = 2,
                                           widths = unit.c(unit(1, "npc") - lwidth, lwidth)))


  showPlotGrid(name, combined, width=3.25, height=6, unit = "in")
  # return gtable invisibly
  invisible(combined)

}


preProcess <- function(directoryName, benchmarkName, hasInput=TRUE){
  #################### No Crash #########################
  readFile <- paste(directoryName, "/", benchmarkName,".csv", sep="")
  res <- read.csv(readFile, stringsAsFactors=F)

  tmpMean <- aggregate(. ~ benchmark + input + numThreads  + deviceKind, data = res, mean)
  print((tmpMean))

  tmpMean$totalTime <- tmpMean$totalTime/1000
  #if(hasInput)
    #p <- ggplot(tmpMean, aes(x=numThreads, y=totalTime)) + geom_line(color="steelblue") + geom_point(color="steelblue") + facet_grid(~input, scales="free_y") + scale_y_continuous("Time Total (s)")+ theme(axis.text.x = element_text(angle = 0)) + scale_x_continuous("Number of Threads", breaks = unique(tmpMean$numThreads)) + scale_color_manual(values=c("#CC6666")) #+ scale_x_continuous("Hosts", trans="log2", breaks=c(1, 4, 16, 64))
  #else
    p <- ggplot(tmpMean, aes(x=numThreads, y=totalTime)) + geom_line(color="steelblue") + geom_point(color="steelblue4") + scale_y_continuous("Time Total (s)")+ theme(axis.text.x = element_text(angle = 0)) + scale_x_continuous("Number of Threads", breaks = unique(tmpMean$numThreads)) + scale_color_manual(values=c("#CC6666")) #+ scale_x_continuous("Hosts", trans="log2", breaks=c(1, 4, 16, 64))

  outFileName <- paste(benchmarkName, "_totalTime", sep="")
  showPlot(outFileName, p, width=3.6, height=5.25, unit = "in")
}

preProcess("./", "barneshut")
preProcess("./", "bc-async")
preProcess("./", "bc-outer")
preProcess("./", "bfs")
preProcess("./", "boruvka")
preProcess("./", "connectedcomponents")
preProcess("./", "delaunaytriangulation")
preProcess("./", "dmr")
preProcess("./", "gmetis")
preProcess("./", "independentset")
preProcess("./", "matrixcompletion")
preProcess("./", "mcm")
preProcess("./", "pagerank-pull")
preProcess("./", "pagerank-push")
preProcess("./", "preflowpush")
preProcess("./", "pta")
preProcess("./", "sssp")
preProcess("./", "surveypropagation")
preProcess("./", "triangles-edge")
preProcess("./", "triangles-node")
