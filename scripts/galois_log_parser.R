#!/usr/bin/env Rscript

#######################################################
# Author: Gurbinder Gill
# Email:  gill@cs.utexas.edu
# Date:   Oct 8, 2017
######################################################
library("optparse")
library('data.table')

####START: @function to parse commadline##################
# Parses the command line to get the arguments used
parseCmdLine <- function (logData, isSharedMemGaloisLog) {
  cmdLineRow <- subset(logData, CATEGORY == "CommandLine"& TOTAL_TYPE != "HostValues")

  ## Distributed has extra column: HostID
  if(isTRUE(isSharedMemGaloisLog))
    cmdLine <- substring(cmdLineRow[,5], 0)
  else
    cmdLine <- substring(cmdLineRow[,6], 0)

  cmdLineSplit = strsplit(cmdLine, "\\s+")[[1]]

  ## First postitional argument is always name of the executable
  ### WORKING: split the exePath name found at the position 1 of the argument list and split on "/".
  exePathSplit <- strsplit(cmdLineSplit[1], "/")[[1]]
  benchmark <- exePathSplit[length(exePathSplit)]

  ## subset the threads row from the table
  numThreads <- (subset(logData, CATEGORY == "Threads"& TOTAL_TYPE != "HostValues"))$TOTAL

  ## subset the input row from the table
  inputPath <- (subset(logData, CATEGORY == "Input"& TOTAL_TYPE != "HostValues"))$TOTAL
  inputPathSplit <- strsplit(inputPath[[1]], "/")[[1]]
  input <- inputPathSplit[length(inputPathSplit)]
  ### This is to remore the extension for example .gr or .sgr
  inputsplit <- strsplit(input, "[.]")[[1]]
  if(length(inputsplit) > 1) {
    input <- inputsplit[1]
  }

 if(isTRUE(isSharedMemGaloisLog)){
   returnList <- list("benchmark" = benchmark, "input" = input, "numThreads" = numThreads)
   return(returnList)
 }

 ## Need more params for distributed galois logs
 numHosts <- (subset(logData, CATEGORY == "Hosts"& TOTAL_TYPE != "HostValues"))$TOTAL

 partitionScheme <- (subset(logData, CATEGORY == "PartitionScheme"& TOTAL_TYPE != "HostValues"))$TOTAL

 runID <- (subset(logData, CATEGORY == "Run_UUID"& TOTAL_TYPE != "HostValues"))$TOTAL

 numIterations <- (subset(logData, CATEGORY == "NUM_ITERATIONS_0"& TOTAL_TYPE != "HostValues"))$TOTAL

 ## returnList for distributed galois log
 returnList <- list("runID" = runID, "benchmark" = benchmark, "input" = input, "partitionScheme" = partitionScheme, "hosts" = numHosts , "numThreads" = numThreads, "iterations" = numIterations)
 return(returnList)
}
#### END: @function to parse commadline ##################

#### START: @function to values of timers for shared memory galois log ##################
# Parses to get the timer values
getTimersShared <- function (logData) {
 ##XXX NULL should not be a string
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
 totalTime <- (subset(logData, CATEGORY == "TIMER_TOTAL" & TOTAL_TYPE != "HostValues")$TOTAL)
 print(paste("totalTime:", totalTime))

 ## Taking mean of all the runs
 totalTimeExecMean <- mean(as.numeric(subset(logData, grepl("TIMER_[0-9]+", CATEGORY) & TOTAL_TYPE != "HostValues")$TOTAL))
 print(paste("totalTimeExecMean:", totalTimeExecMean))

 ## To get the name of benchmark to be used with other queries to get right timers.
 ### It assumes that there will always with TIMER_0 with REGION name as benchmark
 ### name used with other queries.
 benchmarkRegionName <- subset(logData, CATEGORY == "TIMER_0" & TOTAL_TYPE != "HostValues")$REGION
 print(paste("benchmark:", benchmarkRegionName))

 ## Number of runs
 numRuns <- as.numeric((subset(logData, CATEGORY == "Runs" & TOTAL_TYPE != "HostValues"))$TOTAL)
 print(paste("numRuns:", numRuns))

 ## Total compute time (galois::do_alls)
 computeTimePerIter <- numeric()
 for(i in 1:(numRuns)) {
   j = i - 1 #Vectors are 1 indexed in r
   computeTimeRows <- subset(logData, grepl(paste("^", benchmarkRegionName, "_", j, "_[0-9]+", sep=""), REGION) & TOTAL_TYPE != "HostValues")$TOTAL
   #computeTimeRows <- subset(logData, grepl("^BFS_0_0", REGION) & TOTAL_TYPE != "HostValues")
   if(!is.null(computeTimeRows)){
     computeTimePerIter[i] <- sum(as.numeric(computeTimeRows))
   }
 }
 computeTimeMean <- (mean(computeTimePerIter))
 print(paste("computeTimeMean:", computeTimeMean))

 ##Total sync time.
 syncTimePerIter <- numeric()
 for(i in 1:(numRuns)) {
   j = i - 1 #Vectors are 1 indexed in r
   syncTimeRows <- subset(logData, grepl(paste("SYNC_", benchmarkRegionName, "_", j, "_[0-9]+", sep=""), CATEGORY) & TOTAL_TYPE != "HostValues")$TOTAL
   if(!is.null(syncTimeRows)){
     syncTimePerIter[i] <- sum(as.numeric(syncTimeRows))
   }
 }
 syncTimeMean <- (mean(syncTimePerIter))
 print(paste("syncTimeMean", syncTimeMean))


 ## Mean time spent in the implicit barrier: DGReducible
 barrierTimePerIter <- numeric()
 for(i in 1:(numRuns)) {
  j = i - 1 #Vectors are 1 indexed in r
  barrierTimeRows <- subset(logData, REGION =="DGReducible" & grepl(paste( "REDUCE_DGACCUM_", j, "_[0-9]+", sep=""), CATEGORY) & TOTAL_TYPE != "HostValues")$TOTAL
  if(!is.null(barrierTimeRows)){
    barrierTimePerIter[i] <- sum(as.numeric(barrierTimeRows))
  }
 }
 barrierTimeMean <- (mean(barrierTimePerIter))
 print(paste("barrierTimeMean:", barrierTimeMean))

 ## Total bytes sent in reduce and broadcast phase in run 0.
 ### Same number of bytes are being sent in all the runs.
 syncBytes <- sum(as.numeric(subset(logData, grepl(paste("[REDUCE|BROADCAST]_SEND_BYTES_", benchmarkRegionName, "_0_[0-9]+", sep=""), CATEGORY)& TOTAL_TYPE != "HostValues")$TOTAL))
 print(paste("syncBytes:", syncBytes))

 ##Graph construction time
 graphConstructTime <- subset(logData, CATEGORY == "TIME_GRAPH_CONSTRUCT" & TOTAL_TYPE != "HostValues")$TOTAL
 print(paste("graphConstructTime:", graphConstructTime))

 ## Replication factor
 replicationFactor <- subset(logData, CATEGORY == "REPLICATION_FACTOR_0_0" & TOTAL_TYPE != "HostValues")$TOTAL
 print(paste("replicationFactor:", replicationFactor))

 ## Communication memory usage: Max and Min.
 communicationMemUsageMax = as.numeric(subset(logData, CATEGORY == "COMMUNICATION_MEM_USAGE_MAX" & TOTAL_TYPE == "HMAX")$TOTAL)
 communicationMemUsageMin = as.numeric(subset(logData, CATEGORY == "COMMUNICATION_MEM_USAGE_MIN" & TOTAL_TYPE == "HMIN")$TOTAL)

 if(identical(communicationMemUsageMax, numeric(0)) || identical(communicationMemUsageMin, numeric(0))){
   communicationMemUsageMax = 0
   communicationMemUsageMin = 0
   print("Printing Memory usage counter not present.")
 }

 returnList <- list("replicationFac" = replicationFactor, "totalTime" = totalTime, "totalTimeExec" = totalTimeExecMean, "computeTime" = computeTimeMean, "syncTime" = syncTimeMean, "barrierTime" = barrierTimeMean, "syncBytes" = syncBytes, "graphConstructTime"= graphConstructTime, "communicationMemUsageMax" = communicationMemUsageMax, "communicationMemUsageMin" = communicationMemUsageMin)
 return(returnList)
}
#### END: @function to values of timers for distributed memory galois log ##################

#### START: @function to comute per iteration RSD of compute time. ##################
# Parses to get the timer values
computeRSD <- function (logData, paramList, output) {
  numIter = as.numeric(paramList["iterations"])

  benchmarkRegionName <- subset(logData, CATEGORY == "TIMER_0" & TOTAL_TYPE != "HostValues")$REGION
  print(paste("benchmark:", benchmarkRegionName))

  ## Number of runs
  numRuns <- as.numeric((subset(logData, CATEGORY == "Runs" & TOTAL_TYPE != "HostValues"))$TOTAL)
  print(paste("numRuns:", numRuns))

  output_rsd_file <- paste(output, "_computeRSD", sep="")



  ## Doing 1st iteration separately to see if new file is to be created or if file already exists.
  for(r in 0:(numRuns - 1)){
    computeTimeRows <- subset(logData, grepl(paste("^", benchmarkRegionName, "_", r, "_", 0, sep=""), REGION) & TOTAL_TYPE == "HostValues")$TOTAL
    if(!identical(computeTimeRows, character(0))){
      print(computeTimeRows)
      computeTimePerHostArr <- (as.numeric(strsplit(computeTimeRows, ";")[[1]]))
      sd <- sd(computeTimePerHostArr)
      mean <- mean(computeTimePerHostArr)
      rsd <- round((sd/mean)*100, digits = 2)
      rsdList <- list("run" = r, "iter" = 0, "sd" = sd, "mean" = mean , "rsd" = rsd)
      outDataList <- append(paramList, rsdList)
      if(!file.exists(output_rsd_file)){
        print(paste(output_rsd_file, "Does not exist. Creating new file"))
        write.csv(as.data.frame(outDataList), file=output_rsd_file, row.names=F, quote=F)
      } else {
        print(paste("Appending data to the existing file", output_rsd_file))
        write.table(as.data.frame(outDataList), file=output_rsd_file, row.names=F, col.names=F, quote=F, append=T, sep=",")
      }
      print(rsd)
    }
  }

  for(i in 1:(numIter - 1)) {
    for(r in 0:(numRuns - 1)){
      print(i)
      computeTimeRows <- subset(logData, grepl(paste("^", benchmarkRegionName, "_", r, "_", i, sep=""), REGION) & TOTAL_TYPE == "HostValues")$TOTAL
      if(!identical(computeTimeRows, character(0))){
        computeTimePerHostArr <- (as.numeric(strsplit(computeTimeRows, ";")[[1]]))
        sd <- sd(computeTimePerHostArr)
        mean <- mean(computeTimePerHostArr)
        rsd <- round((sd/mean)*100, digits = 2)
        rsdList <- list("run" = r, "iter" = i, "sd" = sd, "mean" = mean , "rsd" = rsd)
        outDataList <- append(paramList, rsdList)
        write.table(as.data.frame(outDataList), file=output_rsd_file, row.names=F, col.names=F, quote=F, append=T, sep=",")
        print(rsd)
      }
    }
  }
}

#### START: @function entry point for galois log parser ##################
galoisLogParser <- function(input, output, isSharedMemGaloisLog, isComputeRSD) {
  logData <- read.csv(input, stringsAsFactors=F,strip.white=T)

  if(isTRUE(isSharedMemGaloisLog)){
    print("Parsing commadline")
    paramList <- parseCmdLine(logData, T)
    print("Parsing timers for shared memory galois log")
    timersList <- getTimersShared(logData)
  }
  else{
    print("Parsing commadline")
    paramList <- parseCmdLine(logData, F)
    print("Parsing timers for distributed memory galois log")
    if(isTRUE(isComputeRSD)){
      computeRSD(logData, paramList, output)
    }
    else{
      timersList <- getTimersDistributed(logData)
    }
  }

  ## if computing RSD then normal stats are not printed
  if(isTRUE(!isComputeRSD)){
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

#############################################
##  Commandline options.
############################################
option_list = list(
                     make_option(c("-i", "--input"), action="store", default=NA, type='character',
                                               help="name of the input file to parse"),
                     make_option(c("-o", "--output"), action="store", default=NA, type='character',
                                               help="name of the output file to store output"),
                     make_option(c("-s", "--sharedMemGaloisLog"), action="store_true", default=FALSE,
                                               help="Is it a shared memory Galois log? If -s is not used, it will be treated as a distributed Galois log [default %default]"),
                     make_option(c("-r", "--relativeStandardDeviation"), action="store_true", default=FALSE,
                                               help="If want to compute the RSD of per iteration compute time[default %default]")

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
  galoisLogParser(opt$i, opt$o, opt$s, opt$r)
}

##################### END #####################
