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

if (F) {
  # For comparing multiple variants at once assuming that "Kind"
  # is a column that organizes the different variants
  T1 <- subset(df, Threads == 1)[,c("Kind","Time")]
  colnames(T1) <- c("Kind", "T1")
  T1 <- cast(melt(T1), fun.aggregate=mean)
  df <- join(df, T1, by="Kind")
} else {
  # Simplier case if no variants
  T1 <- mean(subset(df, Threads == 1)$Time)
}
qplot(Threads, Time, data=df) + geom_line() + ylim(0, max(df$Time))
qplot(Threads, T1 / Time, data=df) + geom_smooth()



