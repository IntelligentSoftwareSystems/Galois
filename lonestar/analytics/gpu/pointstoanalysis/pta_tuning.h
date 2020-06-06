/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#pragma once
#define GPU_NAME "Quadro 6000"
#define GPU_VERSION_MAJOR 2
#define GPU_VERSION_MINOR 0
#define RT_VERSION 5050
#define DRV_VERSION 6050

#define DEF_THREADS_PER_BLOCK 480
#define UPDATE_THREADS_PER_BLOCK 992
#define HCD_THREADS_PER_BLOCK 256
#define COPY_INV_THREADS_PER_BLOCK 512
#define STORE_INV_THREADS_PER_BLOCK 352
#define GEP_INV_THREADS_PER_BLOCK 512
static const char* TUNING_PARAMETERS =
    "DEF_THREADS_PER_BLOCK 480\nUPDATE_THREADS_PER_BLOCK "
    "992\nHCD_THREADS_PER_BLOCK 256\nCOPY_INV_THREADS_PER_BLOCK "
    "512\nSTORE_INV_THREADS_PER_BLOCK 352\nGEP_INV_THREADS_PER_BLOCK 512\n";
