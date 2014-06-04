# Generate pretty graphs from csv
#  usage: Rscript report.R <report.csv>

library(ggplot2)
library(reshape2)

Id.Vars <- c("Algo","Kind","Hostname","Threads","CommandLine")

outputfile <- ""
if (interactive()) {
  theme_grey()
  inputfile <- "~/report.csv"
  outputfile <- ""
} else {
  theme_set(theme_bw(base_size=9))
  arguments <- commandArgs(trailingOnly=TRUE)
  if (length(arguments) < 1) {
    cat("usage: Rscript report.R <report.csv> [output.json]\n")
    quit(save = "no", status=1)
  } 
  inputfile <- arguments[1]
  if (length(arguments) > 1) {
    outputfile <- arguments[2]
  }
}

### Replace orig[i] with repl[i] in x
mreplace <- function(x, orig, repl, defaultIndex=1) {
  stopifnot(length(orig) == length(repl))
  indices <- numeric(length(x))
  for (i in 1:length(orig)) {
    indices[grep(orig[i], x, fixed=T)] <- i
  }
  if (sum(indices == 0) > 0) {
    warning("Some entries were unmatched, using default index")
    indices[indices == 0] <- defaultIndex
  }
  return(sapply(indices, function(x) { repl[x] }))
}

printMapping <- function(lhs, rhs) {
  cat("Using following mapping from command line to (Algo,Kind):\n")
  cat(paste("  ", paste(lhs, rhs, sep=" -> ")), sep="\n")
}

### Generate unique and short set of Algo and Kind names from CommandLine
parseAlgo <- function(res) {
  cmds.orig <- sub(" -t \\d+", "", res$CommandLine)
  cmds.uniq <- unique(cmds.orig)

  make <- function(rhs) {
    cmds.new <- mreplace(cmds.orig, cmds.uniq, rhs)
    printMapping(cmds.uniq, rhs)
    parts <- strsplit(cmds.new, "\\s+") 
    res$Algo <- sapply(parts, function(x) { x[1] })
    res$Kind <- sapply(parts, function(x) { ifelse(length(x) > 1, x[2], "") })
    return(res)
  }

  # Try various uniquification patterns until one works
  # Case 1: Extract algo and give kinds unique within an algo
  algos <- sub("^\\S*?((?:\\w|-|\\.)+)\\s.*$", "\\1", cmds.uniq, perl=T)
  kinds <- numeric(length(algos))
  dupes <- duplicated(algos)
  version <- 0
  while (sum(dupes) > 0) {
    kinds[dupes] <- version
    dupes <- duplicated(paste(algos, kinds))
    version <- version + 1
  }
  rhs <- unique(paste(algos, kinds))
  if (length(cmds.uniq) == length(rhs)) {
    return(make(rhs))
  }

  # Case 2: XXX/XXX/algo ... kind
  rhs <- unique(sub("^\\S*?(\\w+)\\s.*?((?:\\w|\\.)+)$", "\\1 \\2", cmds.uniq, perl=T))
  if (length(cmds.uniq) == length(rhs)) {
    return(make(rhs))
  }
  
  # Case 3: Make all algos unique
  return(make(1:length(cmds.uniq)))
}

res.raw <- read.csv(inputfile, stringsAsFactors=F)
res <- res.raw[res.raw$CommandLine != "",]
cat(sprintf("Dropped %d empty rows\n", nrow(res.raw) - nrow(res)))
res <- parseAlgo(res)

# Timeouts
if ("Timeout" %in% colnames(res)) {
  timeouts <- !is.na(res$Timeout)
  res[timeouts,]$Time <- res[timeouts,]$Timeout
} else {
  timeouts <- FALSE
}
res$Timeout <- timeouts
cat(sprintf("Timed out rows: %d\n", sum(timeouts)))

# Process partial results
partial <- is.na(res$Time)
res <- res[!partial,]
cat(sprintf("Dropped %d partial runs\n", sum(partial)))

# Drop unused columns
Columns <- sapply(res, is.numeric) | colnames(res) %in% Id.Vars
Columns <- names(Columns)[Columns]
Columns <- grep("\\.null\\.", Columns, value=T, invert=T)
res <- res[,Columns]

summarizeBy <- function(d, f, fun.aggregate=mean, suffix=".Y", merge.with=d, idvars=Id.Vars) {
  m <- dcast(melt(d[,Columns], id.vars=idvars), f, fun.aggregate)
  vars <- all.vars(f)
  merge(merge.with, m, by=vars[-length(vars)], all.x=T, suffixes=c("", suffix))
}

# Replace NAs with zeros
res[is.na(res)] <- 0

# Make factors
res <- data.frame(lapply(res, function(x) if (is.character(x)) { factor(x) } else {x}))

# Take mean of multiple runs
#res <- recast(res, ... ~ variable, mean, id.var=Id.Vars)

# Summarize
res <- summarizeBy(subset(res, Threads==1),
                   Algo + Hostname ~ variable,
                   min, ".Ref", merge.with=res)
res <- summarizeBy(res,
                   Algo + Kind + Hostname + Threads + CommandLine ~ variable,
                   mean, ".Mean")

###
# Might as well generate simplified json output
###
if (outputfile != "") {
  library(plyr)
  library(rjson)
  idvars <- setdiff(Id.Vars, c("Threads","CommandLine"))
  json <- toJSON(dlply(res, idvars, function(x) {
    # Just pick representative command line
    cmd <- x[1,]$CommandLine
    cmd <- sub(" -t \\d+", "", cmd)
    # Drop columns being selected on because they are redundant
    d <- x[,setdiff(Columns, c("CommandLine", idvars))]
    # Sort by num threads
    d <- d[order(d$Threads),]
    list(command=cmd, data=d)
  }))
  cat(json, file=outputfile)
  cat(sprintf("Results in %s\n", outputfile))
} else {
  ggplot(res,
         aes(x=Threads, y=Time.Ref/Time.Mean, color=Kind)) +
         geom_point() + 
         geom_line() + 
         scale_y_continuous("Speedup (rel. to best at t=1)") +
         facet_grid(Hostname ~ Algo, scale="free")

  ggplot(res,
         aes(x=Threads, y=Time.Mean/1000, color=Kind)) +
         geom_point() + 
         geom_line() + 
         scale_y_continuous("Time (s)") +
         facet_grid(Hostname ~ Algo, scale="free")

#ggplot(res,
#       aes(x=Threads, y=Iterations.Mean/Iterations.Ref, color=Kind)) +
#       geom_point() + 
#       geom_line() + 
#       scale_y_continuous("Iterations relative to serial") +
#       facet_grid(Hostname ~ Algo, scale="free")

  cat("Results in Rplots.pdf\n")
}


