/*
 */

/**
 * @file BareMPI.h
 *
 * Contains the BareMPI enum and the command line option that controls bare
 * MPI usage.
 */
#pragma once
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
#include "mpi.h"
#include "llvm/Support/CommandLine.h"

//! Defines types of bare MPI to use
enum BareMPI { 
  noBareMPI, //!< do not use bare MPI; use our network layer
  nonBlockingBareMPI, //!< non blocking bare MPI
  oneSidedBareMPI //!< one sided bare MPI
};

//! Command line option for which kind of bare mpi to use
extern llvm::cl::opt<BareMPI> bare_mpi;
#endif
