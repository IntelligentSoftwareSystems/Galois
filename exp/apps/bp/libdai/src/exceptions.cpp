/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


#include <dai/exceptions.h>


namespace dai {


    std::string Exception::ErrorStrings[NUM_ERRORS] = {
        "Feature not implemented",
        "Assertion failed",
        "Impossible typecast",
        "Requested object not found",
        "Requested belief not available",
        "Unknown ENUM value",
        "Unknown DAI algorithm",
        "Unrecognized parameter estimation method",
        "Unknown Property type",
        "Unknown Property",
        "Malformed Property",
        "Not all mandatory Properties specified",
        "Invalid alias",
        "Cannot read file",
        "Cannot write file",
        "Invalid FactorGraph file",
        "Invalid Evidence file",
        "Invalid Expectation-Maximization file",
        "Quantity not normalizable",
        "Multiple undo levels unsupported",
        "FactorGraph is not connected",
        "Internal error",
        "Runtime error",
        "Out of memory"
    };


}
