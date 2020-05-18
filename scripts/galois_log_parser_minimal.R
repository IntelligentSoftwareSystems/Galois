#!/usr/bin/env Rscript

library("optparse")
library('data.table')

####START: @function to parse commadline##################
# Parses the command line to get the arguments used
parseCmdLine <- function (logData, isSharedMemGaloisLog) {
  cmdLineRow <- subset(logData, CATEGORY == "CommandLine" & STAT_TYPE == "PARAM")

  ## Distributed has extra column: HostID
  if(isTRUE(isSharedMemGaloisLog)){
    cmdLine <- substring(cmdLineRow[,5], 0)
  }
  else
    cmdLine <- substring(cmdLineRow[,6], 0)

  cmdLineSplit = strsplit(cmdLine, "\\s+")[[1]]

  deviceKind = "CPU"
  if(!isTRUE(isSharedMemGaloisLog)){
    ## To check the device kind
    pos = regexpr('-pset', cmdLineSplit)
    deviceKind = ""
    if(sum(pos>0) > 0){
      deviceKind = "GPU"
    } else {
      deviceKind = "CPU"
    }
  }

  ## First postitional argument is always name of the executable
  ### WORKING: split the exePath name found at the position 1 of the argument list and split on "/".
  exePathSplit <- strsplit(cmdLineSplit[1], "/")[[1]]
  benchmark <- exePathSplit[length(exePathSplit)]

  ## subset the threads row from the table
  numThreads <- (subset(logData, CATEGORY == "Threads" & TOTAL_TYPE != "HostValues"))$TOTAL

  input = "noInput"
  ## subset the input row from the table
  inputPath <- (subset(logData, CATEGORY == "Input" & STAT_TYPE == "PARAM"))$TOTAL
  print(inputPath)
  if(!identical(inputPath, character(0))){
    inputPathSplit <- strsplit(inputPath, "/")[[1]]
    input <- inputPathSplit[length(inputPathSplit)]
  }
  else {
    inputPathSplit <- strsplit(inputPath[[2]], "/")[[1]]
    input <- inputPathSplit[length(inputPathSplit)]
  }
  ### This is to remove the extension for example .gr or .sgr
  inputsplit <- strsplit(input, "[.]")[[1]]
  if(length(inputsplit) > 1) {
    input <- inputsplit[1]
  }
  
  if(isTRUE(isSharedMemGaloisLog)){
    returnList <- list("benchmark" = benchmark, "input" = input, "numThreads" = numThreads, "deviceKind" = deviceKind)
    return(returnList)
  }

 ## Need more params for distributed galois logs
 numHosts <- (subset(logData, CATEGORY == "Hosts"& TOTAL_TYPE != "HostValues"))$TOTAL

 partitionScheme <- (subset(logData, CATEGORY == "PartitionScheme"& TOTAL_TYPE != "HostValues"))$TOTAL

 runID <- (subset(logData, CATEGORY == "Run_UUID"& TOTAL_TYPE != "HostValues"))$TOTAL

 numIterations <- (subset(logData, CATEGORY == "NumIterations_0"& TOTAL_TYPE != "HostValues"))$TOTAL
 #If numIterations is not printed in the log files
 if(identical(numIterations, character(0))){
   numIterations <- 0
 }

 ## returnList for distributed galois log
 returnList <- list("benchmark" = benchmark, "input" = input, "partitionScheme" = partitionScheme, "hosts" = numHosts , "numThreads" = numThreads, "deviceKind" = deviceKind, "iterations" = numIterations)
 return(returnList)
}
#### END: @function to parse commadline ##################

#### START: @function to values of timers for shared memory galois log ##################
# Parses to get the timer values
getTimersShared <- function (logData, benchmark) {
  totalTimeRow <- subset(logData, CATEGORY == "Time" & REGION == "(NULL)")
  totalTime <- totalTimeRow$TOTAL
  print(paste("totalTime:", totalTime))
 returnList <- list("totalTime" = totalTime)
 return(returnList)
}
#### END: @function to values of timers for shared memory galois log ##################

#### START: @function to values of timers for distributed memory galois log ##################
# Parses to get the timer values
getTimersDistributed <- function (logData) {

 ## Total time including the graph construction and initialization
 totalTime <- (subset(logData, CATEGORY == "TimerTotal" & TOTAL_TYPE != "HostValues")$TOTAL)
 print(paste("totalTime:", totalTime))

 ## Taking mean of all the runs
 totalTimeExecMean <- round(mean(as.numeric(subset(logData, grepl("Timer_[0-9]+", CATEGORY) & TOTAL_TYPE != "HostValues")$TOTAL)), digits = 2)
 print(paste("totalTimeExecMean:", totalTimeExecMean))

 ## To get the name of benchmark to be used with other queries to get right timers.
 ### It assumes that there will always with Timer_0 with REGION name as benchmark
 ### name used with other queries.
 benchmarkRegionName <- subset(logData, CATEGORY == "Timer_0" & TOTAL_TYPE != "HostValues")$REGION
 print(paste("benchmark:", benchmarkRegionName))

 ## Number of runs
 numRuns <- as.numeric((subset(logData, CATEGORY == "Runs" & TOTAL_TYPE != "HostValues"))$TOTAL)
 print(paste("numRuns:", numRuns))

 ## Total compute time (galois::do_alls)
 computeTimeMean <- 0
 computeTimeRows <- subset(logData, grepl(paste("^", benchmarkRegionName, "[_]*[[:alpha:]]*_[0-9]+", sep=""), REGION) & CATEGORY == "Time" & TOTAL_TYPE == "HMAX")$TOTAL
 computeTimeMean <- round(sum(as.numeric(computeTimeRows))/numRuns, digits = 2)

 print(paste("computeTimeMean:", computeTimeMean))

 ## Synchronization Time
 syncTimeMean <- 0
 syncTimeRows = subset(logData, grepl(paste("Sync_", benchmarkRegionName, "[_]*[[:alpha:]]*_[0-9]+", sep=""), CATEGORY) & TOTAL_TYPE != "HostValues")$TOTAL
 if(!identical(syncTimeRows, character(0))){
   syncTimeMean <- round(sum(as.numeric(syncTimeRows))/numRuns, digits = 2)
 } 
 print(paste("syncTimeMean", syncTimeMean))


 ## Mean time spent in the implicit barrier: Total - (compute +  sync)
 barrierTimeMean = totalTimeExecMean - (computeTimeMean + syncTimeMean)
 if(barrierTimeMean < 0){
   barrierTimeMean <- 0
 }
 print(paste("barrierTimeMean:", barrierTimeMean))

 ## Total bytes sent in reduce and broadcast phase in run 0.
 ### Same number of bytes are being sent in all the runs.
 syncBytes <- 0
 syncBytes <- sum(as.numeric(subset(logData, grepl(paste("[Reduce|Broadcast]SendBytes_", benchmarkRegionName, "[_]*[[:alpha:]]*_0", sep=""), CATEGORY)& TOTAL_TYPE == "HSUM")$TOTAL))
 print(paste("syncBytes:", syncBytes))

###NOTE: Timer are per source for BC
if(benchmarkRegionName == "BC" | benchmarkRegionName == "MRBC") {
   ## Total number of sources for BC
   numSources <- as.numeric((subset(logData, CATEGORY == "NumSources" & TOTAL_TYPE != "HostValues"))$TOTAL)
   
   totalTimeExecMean <- round(totalTimeExecMean/numSources, digits = 2)
   computeTimeMean <- round(computeTimeMean/numSources, digits = 2)
   syncTimeMean <- round(syncTimeMean/numSources, digits = 2) 
   barrierTimeMean <- round(barrierTimeMean/numSources, digits = 2)
   syncBytes <- round(syncBytes/numSources, digits = 2)
 }

 ##Graph construction time
 graphConstructTime <- subset(logData, CATEGORY == "GraphConstructTime" & TOTAL_TYPE != "HostValues")$TOTAL
 print(paste("graphConstructTime:", graphConstructTime))

 ## Replication factor
 replicationFactor <- subset(logData, CATEGORY == "ReplicationFactor" & TOTAL_TYPE != "HostValues")$TOTAL
 print(paste("replicationFactor:", replicationFactor))
 if(identical(replicationFactor, character(0))){
   replicationFactor <- 0
 }
 returnList <- list("replicationFac" = replicationFactor, "totalTime" = totalTime, "totalTimeExec" = totalTimeExecMean, "computeTime" = computeTimeMean, "syncTime" = syncTimeMean, "barrierTime" = barrierTimeMean, "syncBytes" = syncBytes, "graphConstructTime"= graphConstructTime)
 return(returnList)
}
#### END: @function to values of timers for distributed memory galois log ##################

#### START: @function entry point for galois log parser ##################
galoisLogParser <- function(input, output, isSharedMemGaloisLog) {
  logData <- read.csv(input, stringsAsFactors=F,strip.white=T)

  printNormalStats = TRUE;
  if(isTRUE(isSharedMemGaloisLog)){
    print("Parsing commadline")
    paramList <- parseCmdLine(logData, T)
    print("Parsing timers for shared memory galois log")
    benchmark = paramList[1]
    timersList <- getTimersShared(logData, benchmark)
  }
  else{
    print("Parsing commadline")
    paramList <- parseCmdLine(logData, F)
    print("Parsing timers for distributed memory galois log")
    timersList <- getTimersDistributed(logData)
    
  }

  if(isTRUE(printNormalStats)){
    outDataList <- append(paramList, timersList)
    if(!file.exists(output)){
      print(paste(output, "Does not exist. Creating new file"))
      write.csv(as.data.frame(outDataList), file=output, row.names=F, quote=F)
    } else {
      print(paste("Appending data to the existing file", output))
      write.table(as.data.frame(outDataList), file=output, row.names=F, col.names=F, quote=F, append=T, sep=",")
    }
  }
}
#### END: @function entry point for shared memory galois log ##################

#### START: @function entry point for de-duplication of entries ##################
deDupByMean <- function(output) {
  logData <- read.csv(output, stringsAsFactors=F,strip.white=T)
  ## Aggregate results from multiple runs
  logData_agg <- aggregate(. ~ benchmark + input + partitionScheme + 
          hosts + numThreads + deviceKind,
          data = logData, mean)
  write.csv(logData_agg, output, row.names=FALSE, quote=FALSE)
}
#### END: @function entry point for de-duplication of entries ##################


#############################################
##  Commandline options.
#######################################
option_list = list(
                   make_option(c("-i", "--input"), action="store", default=NA, type='character',
                               help="Name of the input file to parse"),
                   make_option(c("-o", "--output"), action="store", default=NA, type='character',
                               help="Name of the output file to store output"),
                   make_option(c("-d", "--duplicate"), action="store_true", default=FALSE,
                               help="Allow duplicate entries. By default takes mean of duplicate entries [default %default]"),
                   make_option(c("-s", "--sharedMemGaloisLog"), action="store_true", default=FALSE,
                               help="Is it a shared memory Galois log? If -s is not used, it will be treated as a distributed Galois log [default %default]")
                   )

opt_parser <- OptionParser(usage = "%prog [options] -i input.log -o output.csv", option_list=option_list)
opt <- parse_args(opt_parser)

if (is.na(opt$i)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file)", call.=FALSE)
} else {
  if (is.na(opt$o)){
    print("Output file name is not specified. Using name ouput.csv as default")
    opt$o <- "output.csv"
  }
  print(opt$g)
  galoisLogParser(opt$i, opt$o, opt$s)
  ## Take mean of the duplicate entries ##
  if(!opt$d){
    deDupByMean(opt$o)
  }
}
##################### END #####################
