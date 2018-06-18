#pragma once

/* this file is included into CUDA code, do not include Galois header files */

// Constants for page Rank Algo.
//! d is the damping factor. Alpha is the prob that user will do a random jump,
//! i.e., 1 - d
static const double alpha = (1.0 - 0.85);

//! maximum relative change until we deem convergence
static const double TOLERANCE = 0.1;

// residual error threshold
static const double ERROR_THRESHOLD = 0.015;