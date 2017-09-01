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
  };

  struct SyncFlag_entry {
    string FIELD_NAME;
    string RW; // read or write
    string AT; // src or dest
  };

  struct FirstIter_struct_entry{
    //pair: type, varName
    vector<pair<string,string>> MEMBER_FIELD_VEC;
    string OPERATOR_RANGE;
    string OPERATOR_ARG;
    string OPERATOR_BODY;
  };

  class InfoClass {
    public:
      map<string, vector<Graph_entry>> graphs_map;
      map<string, vector<getData_entry>> getData_map;
      map<string, vector<getData_entry>> edgeData_map;
      map<string, vector<NodeField_entry>> fieldData_map;
      map<string, vector<ReductionOps_entry>> reductionOps_map;
      map<string, vector<FirstIter_struct_entry>> FirstItr_struct_map;
      map<string, vector<SyncFlag_entry>>syncFlags_map;
  };


