# Generate pretty graphs from csv
#  usage: Rscript report.R <report.csv>

#theme_set(theme_bw())

library(ggplot2)
args <- commandArgs(trailingOnly=TRUE)

if (length(args) < 1) {
  cat("usage: Rscript report-website.R <report.csv> [serial time]\n")
  quit(save = "no", status=1)
} 

df <- read.csv(args[1])

T1 <- mean(subset(df, Threads == 1)$Time)
qplot(Threads, Time, data=df,ylab="Time (ms)") + geom_line() + ylim(0, max(df$Time))
if (length(args) > 1) {
  last_plot() + geom_hline(yintercept=as.numeric(args[2]))
}
ggsave("time.png", width=4, height=3, dpi=100)
qplot(Threads, T1 / Time, data=df,ylab="Self-relative Speedup") + geom_smooth()
ggsave("speedup.png", width=4, height=3, dpi=100)
