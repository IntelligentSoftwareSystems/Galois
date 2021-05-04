#!/usr/bin/env Rscript

#######################################################
# Author: Gurbinder Gill
# Email:  gill@cs.utexas.edu
# Date:   Oct 8, 2017
######################################################
library("optparse")
library('data.table')

convertZeroTosStr <- function(a) {
  if (identical(numeric(0), as.numeric(a)) == 0) {
    a <- as.numeric(a) / 1000
  } else {
    a <- "0"
  }
  return (a)
}

####START: @function to parse commadline##################
# Parses the command line to get the arguments used
parseCmdLine <- function (logData, isSharedMemGaloisLog, graphPassedAsInput) {
   ## Select commandline & param rows
  cmdLineRow <- subset(logData, CATEGORY == "CommandLine" & STAT_TYPE == "PARAM")

  ## Distributed has extra column: HostID
  if(isTRUE(isSharedMemGaloisLog)){
    cmdLine <- substring(cmdLineRow[,5], 0)
  }
  else {
    cmdLine <- substring(cmdLineRow[,6], 0)
  }

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
  if(isTRUE(graphPassedAsInput)){
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

    ### This is to remore the extension for example .gr or .sgr
    inputsplit <- strsplit(input, "[.]")[[1]]
    if(length(inputsplit) > 1) {
      input <- inputsplit[1]
    }
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

 end2endTimer <- (subset(logData, CATEGORY == "Timer_0"& TOTAL_TYPE != "HostValues"))$TOTAL
 end2endTimer <- convertZeroTosStr(end2endTimer)

 aggr_fwd <- (subset(logData, CATEGORY == "AggregateForward"))$TOTAL
 aggr_fwd <- convertZeroTosStr(aggr_fwd)

 aggr_bwd <- (subset(logData, CATEGORY == "AggregateBackward"))$TOTAL
 aggr_bwd <- convertZeroTosStr(aggr_bwd)

 fwd_total <- (subset(logData, CATEGORY == "ForwardPhase"))$TOTAL
 fwd_total <- convertZeroTosStr(fwd_total)

 fwd_xform <- (subset(logData, CATEGORY == "ForwardXForm"))$TOTAL
 fwd_xform <- convertZeroTosStr(fwd_xform)

 bwd_total <- (subset(logData, CATEGORY == "BackwardPhase"))$TOTAL
 bwd_total <- convertZeroTosStr(bwd_total)

 bwd_xform <- (subset(logData, CATEGORY == "BackwardXForm"))$TOTAL
 bwd_xform <- convertZeroTosStr(bwd_xform)

 avg_epoch <- (subset(logData, CATEGORY == "AverageEpochTime"))$TOTAL
 avg_epoch <- convertZeroTosStr(avg_epoch)

 final_accuracy <- (subset(logData, CATEGORY == "FinalTestAccuracy"))$TOTAL

 train_time <- (subset(logData, CATEGORY == "TrainingTime"))$TOTAL
 train_time <- convertZeroTosStr(train_time)

 sync_aggr <- (subset(logData, CATEGORY == "Sync_GraphAggregateSync_0"))$TOTAL
 sync_aggr <- convertZeroTosStr(sync_aggr)

 sync_weight <- (subset(logData, CATEGORY == "Sync_WeightGradientsSum"))$TOTAL
 sync_weight <- convertZeroTosStr(sync_weight)
 
 buff_breserve_time <- (subset(logData, CATEGORY ==
                       "BroadcastExtract_GraphAggregateSync_0"))$TOTAL 
 buff_breserve_time <- convertZeroTosStr(buff_breserve_time)
 buff_bextract_time <- (subset(logData, CATEGORY ==
                       "BroadcastExtractBatch_GraphAggregateSync_0"))$TOTAL 
 buff_bextract_time <- convertZeroTosStr(buff_bextract_time)
 buff_rreserve_time <- (subset(logData, CATEGORY ==
                       "ReduceExtract_GraphAggregateSync_0"))$TOTAL 
 buff_rreserve_time <- convertZeroTosStr(buff_rreserve_time)
 buff_rextract_time <- (subset(logData, CATEGORY ==
                       "ReduceExtractBatch_GraphAggregateSync_0"))$TOTAL 
 buff_rextract_time <- convertZeroTosStr(buff_rextract_time)

 print(input)
 print(partitionScheme)
 print(numHosts)
 ## returnList for distributed galois log
 returnList <- list("RunID" = runID, "Benchmark" = benchmark,
                    "Input" = input, "PartitionScheme" = partitionScheme,
                    "Hosts" = numHosts, "NumThreads" = numThreads,
                    "EndToEndTime" = end2endTimer,
                    "TrainTime" = train_time,
                    "TotalForwardTime" = fwd_total,
                    "ForwardAggregate" = aggr_fwd,
                    "ForwardXform" = fwd_xform,
                    "TotalBackwardTime" = bwd_total,
                    "BackwardAggregate" = aggr_bwd,
                    "BackwardXfrom" = bwd_xform,
                    "AverageEpochTime" = avg_epoch,
                    "FinalTestAccuracy" = final_accuracy,
                    "AggregateSync" = sync_aggr,
                    "Broadcast_buf_reserve" = buff_breserve_time,
                    "Broadcast_buf_extract" = buff_bextract_time,
                    "Reduce_buf_reserve" = buff_rreserve_time,
                    "Reduce_buf_extract" = buff_rreserve_time,
                    "AggregateWeight" = sync_weight,
                    "DeviceKind" = deviceKind)

 print("List")
 print(returnList)
 # Timer is milli-sec unit 
 return(returnList)
}
#### END: @function to parse commadline ##################

#### START: @function entry point for galois log parser ##################
galoisLogParser <- function(input, output) {
  logData <- read.csv(input, stringsAsFactors=F, strip.white=T)

  printNormalStats = TRUE;
  print("Parsing commadline")
  paramList <- parseCmdLine(logData, F, T)
  print("Parsing timers for shared memory galois log")

  ## if computing RSD then normal stats are not printed
  if(isTRUE(printNormalStats)){
    if(!file.exists(output)){
      print(paste(output, "Does not exist. Creating new file"))
      print(as.data.frame(paramList))
      write.csv(as.data.frame(paramList), file=output, row.names=F, quote=F)
    } else {
      print(paste("Appending data to the existing file", output))
      write.table(as.data.frame(paramList), file=output, row.names=F, col.names=F, quote=F, append=T, sep=",")
    }
  }
}
#### END: @function entry point for shared memory galois log ##################

#############################################
##  Commandline options.
#######################################
option_list = list(
                   make_option(c("-i", "--input"), action="store", default=NA, type='character',
                               help="name of the input file to parse"),
                   make_option(c("-o", "--output"), action="store", default=NA, type='character',
                               help="name of the output file parsed")
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
  galoisLogParser(opt$i, opt$o)
}

##################### END #####################
