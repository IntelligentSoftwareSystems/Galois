/** clang analysis plugin for Galois programs: AST handler -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description: Helper functions for compiler plugin.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */


#ifndef _PLUGIN_ANALYSIS_HELPER_FUNCTIONS_H
#define _PLUGIN_ANALYSIS_HELPER_FUNCTIONS_H


#include <sstream>
#include <string>
#include <algorithm>

using namespace std;
/**************************** Helper Functions: start *****************************/
  /* Convert double to string with specified number of places after the decimal
   *    and left padding. */
  template<typename ty>
  std::string prd(const ty x, const int width) {
      stringstream ss;
      ss << fixed << right;
      ss.fill(' ');        // fill space around displayed #
      ss.width(width);     // set  width around displayed #
      ss << x;
      return ss.str();
  }



/*! Center-aligns string within a field of width w. Pads with blank spaces
 *     to enforce alignment. */
std::string center(const string s, const int w) {
  stringstream ss, spaces;
  int padding = w - s.size();                 // count excess room to pad
  for(int i=0; i<padding/2; ++i)
    spaces << " ";
  ss << spaces.str() << s << spaces.str();
  if(padding>0 && padding%2!=0)               // if odd #, add 1 space
    ss << " ";
  return ss.str();
}
std::string center(const bool s, const int w) {
  stringstream ss, spaces;
  int padding = w - 1;                 // count excess room to pad
  for(int i=0; i<padding/2; ++i)
    spaces << " ";
  ss << spaces.str() << s << spaces.str();
  if(padding>0 && padding%2!=0)               // if odd #, add 1 space
    ss << " ";
  return ss.str();
}

// finds and replaces all occurrences
// ensures progress even if replaceWith contains toReplace
void findAndReplace(std::string &str, 
    const std::string &toReplace, 
    const std::string &replaceWith) 
{
  std::size_t pos = 0;
  while (1) {
    std::size_t found = str.find(toReplace, pos);
    if (found != std::string::npos) {
      str = str.replace(found, toReplace.length(), replaceWith);
      pos = found + replaceWith.length();
    } else {
      break;
    }
  }
}




/** Function to split a string with a given delimiter **/
void split(const string& s, char delim, std::vector<string>& elems) {
  stringstream ss(s);
  string item;
  while(std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

/** Check for a duplicate entry in the map **/
template <typename EntryTy>
bool syncPull_reduction_exists(EntryTy entry, vector<EntryTy> entry_vec){
  if(entry_vec.size() == 0){
    return false;
  }
  for(auto it : entry_vec){
    if((it.SYNC_TYPE == "sync_pull") && (it.FIELD_NAME == entry.FIELD_NAME) && (it.FIELD_TYPE == entry.FIELD_TYPE) && (it.NODE_TYPE == entry.NODE_TYPE) && (it.GRAPH_NAME == entry.GRAPH_NAME)){
      return true;
    }
  }
  return false;
}

/** Check for a duplicate entry in the map **/
template <typename EntryTy>
bool syncFlags_entry_exists(EntryTy entry, vector<EntryTy> entry_vec){
  if(entry_vec.size() == 0){
    return false;
  }
  for(auto it : entry_vec){
    if((it.FIELD_NAME == entry.FIELD_NAME) && (it.RW == entry.RW) && (it.AT == entry.AT)){
      return true;
    }
  }
  return false;
}

/** Check for a duplicate entry in the map **/
template <typename EntryTy>
bool newVariable_entry_exists(EntryTy entry, vector<EntryTy> entry_vec){
  if(entry_vec.size() == 0){
    return false;
  }
  for(auto it : entry_vec){
    if((it.FIELD_NAME == entry.FIELD_NAME) && (it.FIELD_TYPE == entry.FIELD_TYPE)){
      return true;
    }
  }
  return false;
}



bool is_vector_type(string inputStr, string& Type, string& Num_ele) {
  /** Check if type is suppose to be a vector **/
  //llvm::outs()<< "inside is_vec : " <<  inputStr << "\n ";
  //std::regex re("([(a-z)]+)[[:space:]&]*(\[[0-9][[:d:]]*\])");
  std::regex re("(.*)(\[[0-9][[:d:]]*\])");
  std::smatch sm;
  if(std::regex_match(inputStr, sm, re)) {
    Type = "std::vector<" + sm[1].str() + ">";
    llvm::outs() << "INSIDE ALSO : " << Type << "\n";
    Num_ele = sm[2].str();
    Num_ele.erase(0,1);
    Num_ele.erase(Num_ele.size() -1);
    return true;
  }
  return false;
}


/** To split string on Dot operator and return num numbered argument **/
template<typename Ty>
string split_dot(Ty memExpr, unsigned num ){
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  string str_field;
  llvm::raw_string_ostream s(str_field);
  /** Returns something like snode.field_name.operator **/
  memExpr->printPretty(s, 0, Policy);

  /** Split string at delims to get the field name **/
  char delim = '.';
  vector<string> elems;
  split(s.str(), delim, elems);

  if(num == 0)
    return elems[0];
  if(num == 1)
    return elems[1];
}

template<typename Ty>
string printPretty(Ty memExpr){
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  string str_field;
  llvm::raw_string_ostream s(str_field);
  memExpr->printPretty(s, 0, Policy);
  return s.str();
}


string global_var_from_local(string localVar){
  std::regex re ("^local_(.*)");
  std::smatch sm;
  if(std::regex_match(localVar, sm, re)){
    return sm[1].str();
  }
  else {
    return "";
  }
}

/** Given reduction op vector, returns if field is present to be sync_pushed **/
// Basically tells if field is modified.
template<typename Ty>
//bool is_Field_Sync_Pushed(std::string name, std::vector<ReductionOps_entry>& vec){
bool is_Field_Sync_Pushed(std::string name, std::vector<Ty>& vec){
  for(auto i : vec){
    if((i.FIELD_NAME == name) && (i.SYNC_TYPE == "sync_push")){
      return true;
    }
  }
  return false;
}

/**This is helper functions used to make first iteration structure for loop transform.**/
template<typename Ty_firstEntry,typename Ty_reductionOps>
//string makeFunctorFirstIter(string orig_functor_name, FirstIter_struct_entry entry, vector<ReductionOps_entry>& redOp_vec){
string makeFunctorFirstIter(string orig_functor_name, Ty_firstEntry entry, vector<Ty_reductionOps>& redOp_vec){
  string functor = "struct FirstItr_" + orig_functor_name + "{\n";

  /** Adding member functions **/
  for(auto memField : entry.MEMBER_FIELD_VEC){
    functor += (memField.first + " " + memField.second + ";\n");
  }

  /** Adding constructors **/
  string constructor = "FirstItr_" + orig_functor_name + "(";
  string initList = "";
  string initList_call = "";
  unsigned num = entry.MEMBER_FIELD_VEC.size();
  for(auto memField : entry.MEMBER_FIELD_VEC){
    --num;
    if(num > 0){
      constructor += ( memField.first + " _" + memField.second + ",");
      initList += (memField.second + "(_" + memField.second + "),");
      string global_var = global_var_from_local(memField.second);
      if(!global_var.empty())
        initList_call += (global_var + ",");
    }
    else {
      constructor += ( memField.first + " _" + memField.second + ")");
      initList += (memField.second + "(_" + memField.second + "){}");
      string  global_var = global_var_from_local(memField.second);
      if(!global_var.empty())
        initList_call += (global_var);
    }
  }

  constructor += (":" + initList);
  functor += (constructor + "\n");

  /** Insert write sets **/
  stringstream SS_write_set;
  for(auto j : redOp_vec){
    if(j.SYNC_TYPE == "sync_push")
      SS_write_set << ", galois::write_set(\"" << j.SYNC_TYPE << "\", \"" << j.GRAPH_NAME << "\", \"" << j.NODE_TYPE << "\", \"" << j.FIELD_TYPE << "\" , \"" << j.FIELD_NAME << "\", \"" << j.VAL_TYPE << "\" , \"" << j.OPERATION_EXPR << "\",  \"" << j.RESETVAL_EXPR << "\")";
    else if(j.SYNC_TYPE == "sync_pull")
      SS_write_set << ", galois::write_set(\"" << j.SYNC_TYPE << "\", \"" << j.GRAPH_NAME << "\", \""<< j.NODE_TYPE << "\", \"" << j.FIELD_TYPE << "\", \"" << j.FIELD_NAME << "\" , \"" << j.VAL_TYPE << "\")";
  }

  //TODO: Hack.. For now I am assuming is range has begin, all nodes with edges to be operated on.
  std::string operator_range = "";
  std::string operator_range_definition = "";
  std::string orig_operator_range = entry.OPERATOR_RANGE;
  //To remove the , in the range.
  orig_operator_range = orig_operator_range.substr(0,orig_operator_range.find(",",0));
  bool onlyOneNode = false;
  if(orig_operator_range.find(std::string("begin")) != std::string::npos){
    operator_range_definition = "\nauto nodesWithEdges = _graph.allNodesWithEdgesRange(); \n";
    operator_range = "nodesWithEdges,";
  } else {
    operator_range_definition = "\n uint32_t __begin, __end;\n if (_graph.isLocal(" + orig_operator_range + ")) {\n __begin = _graph.getLID(" + orig_operator_range + ");\n__end = __begin + 1;\n} else {\n__begin = 0;\n__end = 0;\n}\n";

    operator_range = " _graph.begin() + __begin , _graph.end() + __end,";
    onlyOneNode = true;
    
  }
  /** Adding static go function **/
  //TODO: Assuming graph type is Graph
  string static_go = "void static go(Graph& _graph) {\n";
  static_go += operator_range_definition;
  if(onlyOneNode)
    static_go += "galois::do_all(" + operator_range;
  else
    static_go += "galois::do_all(" + operator_range;
  static_go += "FirstItr_" + orig_functor_name + "{" + initList_call + "&_graph" + "}, ";
  static_go += "galois::loopname(_graph.get_run_identifier(\"" + orig_functor_name + "\").c_str()),";
  if(!onlyOneNode)
    static_go += "\ngalois::steal<true>(),";

  static_go += "\ngalois::timeit()";
  static_go += SS_write_set.str() + ");\n}\n";
  functor += static_go;

  string operator_func = entry.OPERATOR_BODY + "\n";
  functor += operator_func;


  functor += "\n};\n";

  return functor;
}
/**************************** Helper Functions: End *****************************/

/***Global constants ***/
string galois_distributed_accumulator_type = "galois::DGAccumulator<int> ";
string galois_distributed_accumulator_name = "DGAccumulator_accum";


#endif//_PLUGIN_ANALYSIS_HELPER_FUNCTIONS_H
