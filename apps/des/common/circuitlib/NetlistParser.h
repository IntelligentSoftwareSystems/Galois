/*
 * NetlistParser.h
 *
 *  Created on: Jun 23, 2011
 *      Author: amber
 */

#ifndef NETLISTPARSER_H_
#define NETLISTPARSER_H_

#include <vector>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <boost/utility.hpp>

#include <cstring>
#include <cassert>
#include <cstdio>

#include "logicDefs.h"
#include "comDefs.h"

class NetlistTokenizer: public boost::noncopyable {
  // correct way to use is to first check if hasMoreTokens and then call nextToken
  // 
  // need to read one token ahead so because hasMoreTokens is called before nextToken
  //
  // basic algorithm
  //
  // initially currTokPtr = NULL, nextTokPtr = nextTokenInt
  //
  // In nextToken
  //    return string at currTokPtr
  //    currTokPtr = nextTokPtr
  //    nextTokPtr = read next token
  //
  // algorithm for reading next token
  // read next token (with NULL)
  // while next token is null or beginning of comment {
  //    read next line (break out the loop if file has ended)
  //    read first token
  // }
  // create a string and return
  //
  // things to check for
  // - end of file, reading error (in this case getNextLine() should return NULL)
  // 
  // initialization:
  // - initially nextTokPtr should be NULL and this fine because
  // reading nextTok with null should return null;
  
private:
  std::ifstream ifs;

  const char* delim;
  const char* comments;


  char* linePtr;
  char* nextTokPtr;
  char* currTokPtr;

private:
  bool isCommentBegin () const {
    std::string tok(nextTokPtr);
    if (tok.find (comments) == 0) {
      return true;
    } else {
      return false;
    }
  }

  char* getNextLine () {
    // read next line
    std::string currLine;

    if (ifs.eof () || ifs.bad ()) {
      linePtr = NULL;
    }
    else {

      std::getline (ifs, currLine);

      delete[] linePtr;
      linePtr = new char[currLine.size () + 1];
      strcpy (linePtr, currLine.c_str ());
    }

    return linePtr;
  }

  char* readNextToken () {
    nextTokPtr = strtok (NULL, delim);

    while (nextTokPtr == NULL || isCommentBegin ()) {
          linePtr = getNextLine ();
          if (linePtr == NULL) {
            nextTokPtr = NULL;
            break;
          }
          nextTokPtr = strtok (linePtr, delim);
    }
    return nextTokPtr;
  }

public:
  NetlistTokenizer (const char* fileName, const char* delim, const char* comments)
    : ifs(fileName), delim (delim), comments(comments), linePtr (NULL), currTokPtr (NULL)  {

    assert (ifs.good ());      
    nextTokPtr = readNextToken();

  }

  const std::string nextToken () {
    assert (nextTokPtr != NULL);
    
    std::string retval(nextTokPtr);
    nextTokPtr = readNextToken ();

    return retval;
  }

  bool hasMoreTokens () const {
    return !ifs.eof() || nextTokPtr != NULL;
  }
};

/**
 * The Class GateRec stores the data for a specific gate.
 */
struct GateRec {

  /** The name. */
  std::string name;

  /** The net names outputs. */
  std::vector<std::string> outputs;

  /** The net names inputs. */
  std::vector<std::string> inputs;

  /** The delay. */
  SimTime delay;

  /**
   * Adds the output.
   *
   * @param net the net
   */
  void addOutput(const std::string& net) {
    outputs.push_back(net);
  }

  /**
   * Adds the input.
   *
   * @param net the net
   */
  void addInput(const std::string& net) {
    inputs.push_back(net);
  }

  /**
   * Sets the delay.
   *
   * @param delay the new delay
   */
  void setDelay(const SimTime& delay) {
    this->delay = delay;
  }

  /**
   * Sets the name.
   *
   * @param name the new name
   */
  void setName(const std::string& name) {
    this->name = name;
  }

  /**
   * Gets the name.
   *
   * @return the name
   */
  const std::string& getName() const {
    return name;
  }
};

/**
 * The Class NetlistParser parses an input netlist file.
 */
class NetlistParser {
public:
  //following is the list of token separators; characters meant to be ignored
  static const char* DELIM;
  static const char* COMMENTS;

public:
  typedef std::map<std::string, std::vector<std::pair<SimTime, LogicVal> > > StimulusMapType;

private:
  /** The netlist file. */
  const char* netlistFile;

  /** The input names. */
  std::vector<std::string> inputNames;

  /** The output names. */
  std::vector<std::string> outputNames;

  /** The out values. */
  std::map<std::string, LogicVal> outValues;

  /** The input stimulus map has a list of (time, value) pairs for each input. */
  StimulusMapType inputStimulusMap;

  /** The gates. */
  std::vector<GateRec> gates;

  /** The finish time. */
  SimTime finishTime;



private:

  // initializer block
  static const std::set<std::string>  oneInputGates () {
    std::set<std::string> oneInSet;
    oneInSet.insert (toLowerCase (std::string("INV")));
    return oneInSet;
  }

  static const std::set<std::string> twoInputGates () {
    std::set<std::string> twoInSet;
    twoInSet.insert (toLowerCase (std::string ("AND2")));
    twoInSet.insert (toLowerCase (std::string ("OR2")));
    twoInSet.insert (toLowerCase (std::string ("NAND2")));
    twoInSet.insert (toLowerCase (std::string ("NOR2")));
    twoInSet.insert (toLowerCase (std::string ("XOR2")));
    twoInSet.insert (toLowerCase (std::string ("XNOR2")));
    return twoInSet;
  }


  /**
   * Parses the port list.
   *
   * @param tokenizer the tokenizer
   * @param portNames the net names for input/output ports
   */
  static void parsePortList(NetlistTokenizer& tokenizer, std::vector<std::string>& portNames) {
    std::string token = toLowerCase (tokenizer.nextToken ());
    while (token != ("end")) {
      portNames.push_back(token);
      token = toLowerCase (tokenizer.nextToken ());
    }
  }

  /**
   * Parses the out values.
   *
   * @param tokenizer the tokenizer
   * @param outValues the expected out values at the end of the simulation
   */
  static void parseOutValues(NetlistTokenizer& tokenizer, std::map<std::string, LogicVal>& outValues) {
    std::string token = toLowerCase (tokenizer.nextToken ());
    while (token != ("end")) {
      std::string outName = token;
      token = toLowerCase (tokenizer.nextToken ());
      LogicVal value = token[0];
      token = toLowerCase (tokenizer.nextToken ());

      outValues.insert (std::make_pair(outName, value));
    }
  }

  /**
   * Parses the initialization list for all the inputs.
   *
   * @param tokenizer the tokenizer
   * @param inputStimulusMap the input stimulus map
   */
  static void parseInitList(NetlistTokenizer& tokenizer, StimulusMapType& inputStimulusMap) {
    // capture the name of the input signal
    std::string input = toLowerCase (tokenizer.nextToken ());

    std::string token = toLowerCase (tokenizer.nextToken ());

    std::vector<std::pair<SimTime, LogicVal> > timeValList;
    while (token != ("end")) {

      SimTime t = static_cast<SimTime> (atol (token.c_str ())); // SimTime.parseLong(token);

      token = toLowerCase (tokenizer.nextToken ());
      LogicVal v = token[0];

      timeValList.push_back (std::make_pair(t, v));

      token = toLowerCase (tokenizer.nextToken ());
    }

    inputStimulusMap.insert (std::make_pair(input, timeValList));
  }

  /**
   * Parses the actual list of gates
   *
   * @param tokenizer the tokenizer
   * @param gates the gates
   */
  static void parseNetlist(NetlistTokenizer& tokenizer, std::vector<GateRec>& gates) {

    std::string token = toLowerCase (tokenizer.nextToken ());

    while (token != ("end")) {

      if (oneInputGates().count (token) > 0) {

        GateRec g;
        g.setName(token); // set the gate name

        token = toLowerCase (tokenizer.nextToken ()); // output name
        g.addOutput(token);

        token = toLowerCase (tokenizer.nextToken ()); // input
        g.addInput(token);

        // possibly delay, if no delay then next gate or end
        token = toLowerCase (tokenizer.nextToken ());
        if (token[0] == '#') {
          token = token.substr(1);
          SimTime d = static_cast<SimTime> (atol (token.c_str ())); // SimTime.parseLong(token);
          g.setDelay(d);
          gates.push_back (g);
        } else {
          gates.push_back (g);
          continue;
        }
      } else if (twoInputGates().count (token) > 0) {
        GateRec g;
        g.setName(token); // set the gate name

        token = toLowerCase (tokenizer.nextToken ()); // output name
        g.addOutput(token);

        token = toLowerCase (tokenizer.nextToken ()); // input 1
        g.addInput(token);

        token = toLowerCase (tokenizer.nextToken ()); // input 2
        g.addInput(token);

        // possibly delay, if no delay then next gate or end
        token = toLowerCase (tokenizer.nextToken ());
        if (token[0] == '#') {
          token = token.substr(1);
          SimTime d = static_cast<SimTime> (atol (token.c_str ())); // SimTime.parseLong(token);
          g.setDelay(d);
          gates.push_back (g);
        } else {
          gates.push_back (g);
          continue;
        }
      } else {
        std::cerr << "Unknown type of gate " << token << std::endl;
        assert (false);
      }

      //necessary to move forward in the while loop
      token = toLowerCase (tokenizer.nextToken ());
    } // end of while
  }

public:
  /**
   * Instantiates a new netlist parser.
   *
   * @param netlistFile the netlist file
   */
  NetlistParser(const char* netlistFile): netlistFile(netlistFile) {
    parse(netlistFile);
  }

  /*
   * Parsing steps
   * parse input signal names
   * parse output signal names
   * parse finish time
   * parse stimulus lists for each input signal
   * parse the netlist
   *
   */

  /**
   * Parses the netlist contained in fileName.
   *
   * @param fileName the file name
   */
  void parse(const char* fileName) {
    std::cout << "input: reading circuit from file: " << fileName << std::endl;


    NetlistTokenizer tokenizer (fileName, DELIM, COMMENTS);

    std::string token;

    while (tokenizer.hasMoreTokens()) {

      token = toLowerCase (tokenizer.nextToken ());

      if (token == ("inputs")) {
        parsePortList(tokenizer, inputNames);
      } else if (token == ("outputs")) {
        parsePortList(tokenizer, outputNames);
      } else if (token == ("outvalues")) {
        parseOutValues(tokenizer, outValues);
      } else if (token == ("finish")) {
        token = toLowerCase (tokenizer.nextToken ());
        finishTime = static_cast<SimTime> (atol (token.c_str ())); // SimTime.parseLong(token);
      } else if (token == ("initlist")) {
        parseInitList(tokenizer, inputStimulusMap);
      } else if (token == ("netlist")) {
        parseNetlist(tokenizer, gates);
      }
    } // end outer while

  } // end parse()


  /**
   * Gets the finish time.
   *
   * @return the finish time
   */
  const SimTime& getFinishTime() const {
    return finishTime;
  }

  /**
   * Gets the netlist file.
   *
   * @return the netlist file
   */
  const char* getNetlistFile() const {
    return netlistFile;
  }

  /**
   * Gets the input names.
   *
   * @return the input names
   */
  const std::vector<std::string>& getInputNames() const {
    return inputNames;
  }

  /**
   * Gets the output names.
   *
   * @return the output names
   */
  const std::vector<std::string>& getOutputNames() const {
    return outputNames;
  }

  /**
   * Gets the out values.
   *
   * @return the out values
   */ 
  const std::map<std::string, LogicVal>& getOutValues() const {
    return outValues;
  }

  /**
   * Gets the input stimulus map.
   *
   * @return the input stimulus map
   */
  const StimulusMapType& getInputStimulusMap() const {
    return inputStimulusMap;
  }

  /**
   * Gets the gates.
   *
   * @return the gates
   */
  const std::vector<GateRec>& getGates() const {
    return gates;
  }

};


#endif /* NETLISTPARSER_H_ */
