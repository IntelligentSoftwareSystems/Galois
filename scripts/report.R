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
qplot(Threads, Time, data=df) + geom_line() + ylim(0, max(df$Time))
