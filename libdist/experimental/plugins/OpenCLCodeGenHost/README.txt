/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

This is a simple example demonstrating how to use clang's facility for
providing AST consumers using a plugin.

Build the plugin by running `make` in this directory.

Once the plugin is built, you can run it using:
--
Linux:
$ clang -cc1 -load ../../Debug+Asserts/lib/libPrintFunctionNames.so -plugin print-fns some-input-file.c
$ clang -cc1 -load ../../Debug+Asserts/lib/libPrintFunctionNames.so -plugin print-fns -plugin-arg-print-fns help -plugin-arg-print-fns --example-argument some-input-file.c
$ clang -cc1 -load ../../Debug+Asserts/lib/libPrintFunctionNames.so -plugin print-fns -plugin-arg-print-fns -an-error some-input-file.c

Mac:
$ clang -cc1 -load ../../Debug+Asserts/lib/libPrintFunctionNames.dylib -plugin print-fns some-input-file.c
$ clang -cc1 -load ../../Debug+Asserts/lib/libPrintFunctionNames.dylib -plugin print-fns -plugin-arg-print-fns help -plugin-arg-print-fns --example-argument some-input-file.c
$ clang -cc1 -load ../../Debug+Asserts/lib/libPrintFunctionNames.dylib -plugin print-fns -plugin-arg-print-fns -an-error some-input-file.c
