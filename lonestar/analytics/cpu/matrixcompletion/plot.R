library(grid)
library(gridExtra)
library(ggplot2)
library(plyr)
library(reshape2)

theme_set(theme_bw(base_size=16) + 
  theme(legend.background=element_rect(color="white", fill="white"),
        legend.key=element_rect(color="white", fill="white"),
        #legend.margin=unit(0, "lines"),
        #plot.margin=unit(c(0, 0.25, 0, 0), "lines"),
        strip.background=element_rect(color="white", fill="white")))

dd <- read.csv("data.csv", stringsAsFactors=F)
dd <- dd[,!grepl("null", names(dd))]
dd <- subset(dd, !is.na(RMSE1))
dd$Kind <- ""
dd$Kind[grepl("mc-ddn", dd$CommandLine)] <- "galois"
dd$Kind[grepl("nomad", dd$CommandLine)] <- "nomad"
dd$Kind[grepl("collaborative", dd$CommandLine)] <- "graphlab"
dd$Input <- ""
dd$Input[grepl("bgg", dd$CommandLine)] <- "bgg"
dd$Input[grepl("yahoo", dd$CommandLine)] <- "yahoo"
dd$Input[grepl("netflix", dd$CommandLine)] <- "netflix"

g1 <- ddply(dd, .(Input, Kind, Threads), function (d) {
    ldply(grep("GFLOPS", names(d), value=T), function(n) {
      i=sub("GFLOPS(\\d+)", "\\1", n)
      if (any(is.na(d[,n]))) data.frame() else data.frame(I=as.numeric(i), GFLOPS=d[,n])})
})
g2 <- ddply(dd, .(Input, Kind, Threads), function (d) {
    ldply(grep("Elapsed", names(d), value=T), function(n) {
      i=sub("Elapsed(\\d+)", "\\1", n)
      if (any(is.na(d[,n]))) data.frame() else data.frame(I=as.numeric(i), Elapsed=d[,n])})
})
g3 <- ddply(dd, .(Input, Kind, Threads), function (d) {
    ldply(grep("RMSE", names(d), value=T), function(n) {
      i=sub("RMSE(\\d+)", "\\1", n)
      if (any(is.na(d[,n]))) data.frame() else data.frame(I=as.numeric(i), RMSE=d[,n])})
})

gg <- merge(g1, merge(g2, g3))
gg <- rbind(gg,
  data.frame(
    Input=rep(c("bgg", "netflix", "yahoo"), 3),
    Kind=c(rep("galois", 3), rep("graphlab", 3), rep("nomad", 3)),
    Threads=20,
    I=0,
    GFLOPS=NA,
    Elapsed=0,
    RMSE=rep(c(70.6059, 3.40143, 61.7137), 3)
  ))
gg$Elapsed <- gg$Elapsed / 1000

g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

errorLevelY <- ddply(subset(gg, Threads==20), .(Input), summarize, intercept=1.1*min(RMSE))
errorLevelX <- ddply(subset(gg, Threads==20), .(Input, Kind), function(d) {
  fn <- approxfun(d$Elapsed, d$RMSE)
  target <- subset(errorLevelY, Input==d$Input[1])$intercept[1]
  opt <- optimize(f=function(x) { (fn(x) - target)^2 }, c(0, 1000))
  data.frame(xintercept=opt$minimum)
})

p1 <- ggplot(subset(gg, Input=="bgg" & Threads==20), aes(x=Elapsed, y=RMSE, color=Kind)) +
  geom_line(size=1) +
  scale_x_continuous("") +
  scale_y_continuous("") +
  scale_color_brewer(type="qual") +
  geom_abline(data=subset(errorLevelY, Input=="bgg"), aes(intercept=intercept, slope=0)) +
  geom_vline(data=subset(errorLevelX, Input=="bgg"),
             linetype="dashed",
             aes(xintercept=xintercept, color=Kind)) +
  geom_text(data=subset(errorLevelX, Input=="bgg"),
            color="black",
            aes(x=xintercept, y=0, label=sprintf("%.0f", xintercept), vjust = -1, hjust = 0)) +
  facet_wrap(~Input, scale="free") +
  theme(legend.position="bottom") +
  coord_cartesian(xlim=c(0, 500), ylim=c(0, 15))
p2 <- ggplot(subset(gg, Input=="netflix" & Threads==20), aes(x=Elapsed, y=RMSE, color=Kind)) +
  geom_line(size=1) +
  scale_x_continuous("") +
  scale_y_continuous("") +
  scale_color_brewer(type="qual") +
  geom_abline(data=subset(errorLevelY, Input=="netflix"), aes(intercept=intercept, slope=0)) +
  geom_vline(data=subset(errorLevelX, Input=="netflix"),
             linetype="dashed",
             aes(xintercept=xintercept, color=Kind)) +
  geom_text(data=subset(errorLevelX, Input=="netflix"),
            color="black",
            aes(x=xintercept, y=0.75, label=sprintf("%.0f", xintercept), vjust = -1, hjust = 0)) +
  facet_wrap(~Input, scale="free") +
  theme(legend.position="none") +
  coord_cartesian(xlim=c(0, 200), ylim=c(0.75, 1.5))
p3 <- ggplot(subset(gg, Input=="yahoo" & Threads==20), aes(x=Elapsed, y=RMSE, color=Kind)) +
  geom_line(size=1) +
  scale_x_continuous("") +
  scale_y_continuous("") +
  scale_color_brewer(type="qual") +
  geom_abline(data=subset(errorLevelY, Input=="yahoo"), aes(intercept=intercept, slope=0)) +
  geom_vline(data=subset(errorLevelX, Input=="yahoo"),
             linetype="dashed",
             aes(xintercept=xintercept, color=Kind)) +
  geom_text(data=subset(errorLevelX, Input=="yahoo"),
            color="black",
            aes(x=xintercept, y=c(18, 0, 17), label=sprintf("%.0f", xintercept), vjust = -1, hjust = 0)) +
  facet_wrap(~Input, scale="free") +
  theme(legend.position="none") +
  coord_cartesian(xlim=c(0, 700), ylim=c(17, 30))
plegend <- g_legend(p1)
pp <- arrangeGrob(
  arrangeGrob(p1 + theme(legend.position="none"), p2, p3, nrow=1),
  plegend, nrow=2, heights=c(1, 0.1))

ggplot(gg, aes(x=as.factor(Threads), y=GFLOPS, color=Kind)) + 
  geom_boxplot(position="identity", outlier.colour = NULL) +
  scale_color_brewer(type="qual") +
  scale_x_discrete("Threads") +
  facet_wrap(~Input, scale="free")


dd <- read.csv("/net/peltier/workspace/ddn/build/default/tyahoo-transpose.degreehist")
dd <- read.csv("/net/peltier/workspace/ddn/build/default/tyahoo.degreehist")
dd <- subset(dd, Degree != 0)
mod <- nls(Count ~ exp(a + b * Degree), data=dd, start = list(a=0, b=0))
dd$Kind <- "O"
pp <- data.frame(Hist=0, Degree=dd$Degree, Count=predict(mod, list(x=dd$Degree)), Kind="P")
dd <- rbind(pp, dd)
ggplot(dd, aes(x=Degree, y=Count, color=Kind)) +
  geom_point() +
  scale_x_log10()
