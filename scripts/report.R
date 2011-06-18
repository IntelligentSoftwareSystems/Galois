# Generate pretty graphs from csv
#  usage: Rscript report.R <report.csv>

#theme_set(theme_bw())

library(ggplot2)
args <- commandArgs(trailingOnly=TRUE)

if (length(args) < 1) {
  cat("usage: Rscript report.R <report.csv>\n")
  quit(save = "no", status=1)
} 

df <- read.csv(args[1])
t1 <- mean(subset(df, Threads == 1)$Time)
qplot(Threads, Time, data=df) + geom_smooth() + ylim(0, max(df$Time))
qplot(Threads, t1 / Time, data=df) + geom_smooth()
