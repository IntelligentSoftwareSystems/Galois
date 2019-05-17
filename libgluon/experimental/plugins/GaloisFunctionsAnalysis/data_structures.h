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

/*
 * Desc: This file contains data structures used in the compiler.
 *
 * Author : Gurbinder Gill (gurbinder533@gmail.com)
 */

#include <vector>

struct getData_entry {
  string VAR_NAME;
  string VAR_TYPE;
  string SRC_NAME;
  string RW_STATUS;
  bool IS_REFERENCE;
  bool IS_REFERENCED;
  string GRAPH_NAME;
};

struct Graph_entry {
  string GRAPH_NAME;
  string NODE_TYPE;
  string EDGE_TYPE;
};

struct NodeField_entry {
  string VAR_NAME;
  string VAR_TYPE;
  string FIELD_NAME;
  string NODE_NAME;
  string NODE_TYPE;
  string RW_STATUS;
  bool IS_REFERENCE;
  bool IS_REFERENCED;
  string GRAPH_NAME;
  string RESET_VALTYPE;
  string RESET_VAL;
  string SYNC_TYPE;
  bool IS_VEC;
};

struct ReductionOps_entry {
  string NODE_TYPE;
  string NODE_NAME;
  string FIELD_NAME;
  string FIELD_TYPE;
  string VAL_TYPE;
  string OPERATION_EXPR;
  string RESETVAL_EXPR;
  string GRAPH_NAME;
  string SYNC_TYPE;
  string READFLAG;
  string WRITEFLAG;
  bool IS_RESET;
  bool IS_REDUCTION;
  string MODIFIED_IN_EDGE_LOOP; /** INSIDE or OUTSIDE **/
};

struct SyncFlag_entry {
  string VAR_NAME;
  string FIELD_NAME;
  string RW; // read or write
  string AT; // src or dest
  bool IS_RESET;
};

struct FirstIter_struct_entry {
  // pair: type, varName
  vector<pair<string, string>> MEMBER_FIELD_VEC;
  string OPERATOR_RANGE;
  string OPERATOR_ARG;
  string OPERATOR_BODY;
};

struct New_variable_entry {
  string FIELD_NAME;
  string FIELD_TYPE;
  string INIT_VAL;
};

class InfoClass {
public:
  map<string, vector<Graph_entry>> graphs_map;
  map<string, vector<getData_entry>> getData_map;
  map<string, vector<getData_entry>> edgeData_map;
  map<string, vector<NodeField_entry>> fieldData_map;
  map<string, vector<ReductionOps_entry>> reductionOps_map;
  map<string, vector<FirstIter_struct_entry>> FirstItr_struct_map;
  map<string, vector<SyncFlag_entry>> syncFlags_map;
  map<string, vector<New_variable_entry>> newVariables_map;
};
