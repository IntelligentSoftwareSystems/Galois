/** NetlistParser reads a circuit netlist containing logic gates and wires etc  -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *  Created on: Jun 23, 2011
 *  
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef DES_NETLISTPARSER_H_
#define DES_NETLISTPARSER_H_

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

#include "comDefs.h"
#include "logicDefs.h"
#include "LogicFunctions.h"
#include "LogicGate.h"
#include "OneInputGate.h"
#include "TwoInputGate.h"
#include "BasicPort.h"


namespace des {

/**
 * NetlistTokenizer is a simple string tokenizer, which 
 * usings C strtok () function from <pre>cstring</pre>
 */

class NetlistTokenizer: public boost::noncopyable {
  /**
   * correct way to use is to first check if hasMoreTokens and then call nextToken
   * 
   * need to read one token ahead so because hasMoreTokens is called before nextToken
   *
   * basic algorithm
   *
   * initially currTokPtr = NULL, nextTokPtr = nextTokenInt
   *
   * In nextToken
   *    return string at currTokPtr
   *    currTokPtr = nextTokPtr
   *    nextTokPtr = read next token
   *
   * algorithm for reading next token
   * read next token (with NULL)
   * while next token is null or beginning of comment {
   *    read next line (break out the loop if file has ended)
   *    read first token
   * }
   * create a string and return
   *
   * things to check for
   * - end of file, reading error (in this case getNextLine() should return NULL)
   * 
   * initialization:
   * - initially nextTokPtr should be NULL and this fine because
   * reading nextTok with null should return null;
   *
   */
  
private:
  /** file handle for input stream */
  std::ifstream ifs;

  /** what characters mark end of a token */
  const char* delim;

  /** string representing beginning of a line comment */
  const char* comments;


  /** current line read from the file */
  char* linePtr;

  /** ptr to next token */
  char* nextTokPtr;

  /** ptr to current token, returned on  a call to nextToken () */
  //char* currTokPtr;

private:

  /** 
   * @returns true if nextTokPtr starts with comment pattern
   */
  bool isCommentBegin () const {
    std::string tok(nextTokPtr);
    if (tok.find (comments) == 0) {
      return true;
    } else {
      return false;
    }
  }

  /** 
   * read next line from the file
   * and return it as a C string
   */
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

  /**
   * read next token as a C string
   */
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
  /**
   * Constructor
   * 
   * @param fileName: the file to read from
   * @param delim: a string containing characters that mark end of a token
   * @param comments: a string that contains beginning of a comment
   */
  NetlistTokenizer (const char* fileName, const char* delim, const char* comments)
    : ifs(fileName), delim (delim), comments(comments), linePtr (NULL)  {

    if (!ifs.good ()) {
      std::cerr << "Failed to open this file for reading: " << fileName << std::endl; 
      abort ();
    }
    nextTokPtr = readNextToken();

  }

  /**
   * returns the next token from the file
   */
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
 * The Class NetlistParser parses an input netlist file.
 */
class NetlistParser {
public:

  /** following is the list of token separators; characters meant to be ignored */
  static const char* DELIM;

  /**  beginning of a comment string */
  static const char* COMMENTS;

public:
  typedef std::map<std::string, std::vector<std::pair<SimTime, LogicVal> > > StimulusMapType;

private:


  /** The netlist file. */
  const std::string& netlistFile;

  /** The input names. */
  std::vector<std::string> inputNames;

  /** input ports */
  std::vector<BasicPort*> inputPorts;

  /** The output names. */
  std::vector<std::string> outputNames;

  /** output ports */
  std::vector<BasicPort*> outputPorts;

  /** The out values. */
  std::map<std::string, LogicVal> outValues;

  /** The input stimulus map has a list of (time, value) pairs for each input. */
  StimulusMapType inputStimulusMap;

  /** The gates. */
  std::vector<LogicGate*> gates;

  /** The finish time. */
  SimTime finishTime;



private:
  /**
   * A mapping from string name (in the netlist) to functor that implements
   * the corresponding functionality. Helps in initialization
   */
  static const std::map<std::string, OneInputFunc* >& oneInputGates () {
    static std::map<std::string, OneInputFunc*>  oneInMap;
    oneInMap.insert(std::make_pair (toLowerCase ("INV"), new INV()));
    return oneInMap;
  }

  /**
   * A mapping from string name (in the netlist) to functor that implements
   * the corresponding functionality. Helps in initialization
   */
  static const std::map<std::string, TwoInputFunc*>& twoInputGates () {
    static std::map<std::string, TwoInputFunc*> twoInMap;
    twoInMap.insert(std::make_pair (toLowerCase ("AND2") , new AND2()));
    twoInMap.insert(std::make_pair (toLowerCase ("NAND2") , new NAND2()));
    twoInMap.insert(std::make_pair (toLowerCase ("OR2") , new OR2()));
    twoInMap.insert(std::make_pair (toLowerCase ("NOR2") , new NOR2()));
    twoInMap.insert(std::make_pair (toLowerCase ("XOR2") , new XOR2()));
    twoInMap.insert(std::make_pair (toLowerCase ("XNOR2") , new XNOR2()));
    return twoInMap;
  }


  /**
   * Parses the port list i.e. inputs and outputs
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

  static const char* getInPrefix () { return "in_"; }

  static const char* getOutPrefix () { return "out_"; }

  void createInputPorts () {
    for (std::vector<std::string>::const_iterator i = inputNames.begin (), ei = inputNames.end (); i != ei; ++i) {
      const std::string& out = *i;
      std::string in = getInPrefix() + out;
      inputPorts.push_back (new BasicPort (out, in));
    }
  }

  void createOutputPorts () {
    for (std::vector<std::string>::const_iterator i = outputNames.begin (), ei = outputNames.end (); i != ei; ++i) {
      const std::string& in = *i;
      std::string out = getOutPrefix() + in;
      outputPorts.push_back (new BasicPort (out, in));
    }
  }

  /**
   * Parses the out values, which are the expected values of the circuit outputs at the end of
   * simulation
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

      SimTime t(atol (token.c_str ())); // SimTime.parseLong(token);

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
  static void parseNetlist(NetlistTokenizer& tokenizer, std::vector<LogicGate*>& gates) {

    std::string token = toLowerCase (tokenizer.nextToken ());

    while (token != ("end")) {

      if (oneInputGates().count (token) > 0) {

        const OneInputFunc* func = (oneInputGates ().find (token))->second;

        std::string outputName = toLowerCase (tokenizer.nextToken ()); // output name

        std::string inputName = toLowerCase (tokenizer.nextToken ()); // input


        OneInputGate* g = new OneInputGate (*func, outputName, inputName);
        gates.push_back (g);

        // possibly delay, if no delay then next gate or end
        token = toLowerCase (tokenizer.nextToken ());
        if (token[0] == '#') {
          token = token.substr(1);
          SimTime d(atol (token.c_str ())); // SimTime.parseLong(token);
          g->setDelay(d);

        } else {
          continue;
        }
      } else if (twoInputGates().count (token) > 0) {

        const TwoInputFunc* func = (twoInputGates ().find (token))->second;

        std::string outputName = toLowerCase (tokenizer.nextToken ()); // output name
        std::string input1Name = toLowerCase (tokenizer.nextToken ()); // input 1

        std::string input2Name = toLowerCase (tokenizer.nextToken ()); // input 2

        TwoInputGate* g = new TwoInputGate (*func, outputName, input1Name, input2Name);
        gates.push_back (g);

        // possibly delay, if no delay then next gate or end
        token = toLowerCase (tokenizer.nextToken ());
        if (token[0] == '#') {
          token = token.substr(1);
          SimTime d(atol (token.c_str ())); // SimTime.parseLong(token);
          g->setDelay(d);

        } else {
          continue;
        }
      } else {
        std::cerr << "Unknown type of gate " << token << std::endl;
        abort ();
      }

      //necessary to move forward in the while loop
      token = toLowerCase (tokenizer.nextToken ());
    } // end of while
  }

  /**
     * Parses the netlist contained in fileName.
     *
     * Parsing steps
     * parse input signal names
     * parse output signal names
     * parse finish time
     * parse stimulus lists for each input signal
     * parse the netlist
     *
     * @param fileName the file name
     */

  void parse(const std::string& fileName) {
    std::cout << "input: reading circuit from file: " << fileName << std::endl;


    NetlistTokenizer tokenizer (fileName.c_str (), DELIM, COMMENTS);

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
        finishTime = SimTime (atol (token.c_str ())); // SimTime.parseLong(token);
      } else if (token == ("initlist")) {
        parseInitList(tokenizer, inputStimulusMap);
      } else if (token == ("netlist")) {
        parseNetlist(tokenizer, gates);
      }
    } // end outer while

    createInputPorts ();
    createOutputPorts ();
  } // end parse()


  void destroy () {
    destroyVec (gates);
  }

public:
  /**
   * Instantiates a new netlist parser.
   *
   * @param netlistFile the netlist file
   */
  NetlistParser(const std::string& netlistFile): netlistFile(netlistFile) {
    parse(netlistFile);
  }


  ~NetlistParser () {
    destroy ();
  }




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
  const std::string& getNetlistFile() const {
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
   *
   * @return input ports vector
   */
  const std::vector<BasicPort*>& getInputPorts () const { 
    return inputPorts;
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
   *
   * @return output ports vector
   */
  const std::vector<BasicPort*>& getOutputPorts () const {
    return outputPorts;
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
  const std::vector<LogicGate*>& getGates() const {
    return gates;
  }

};


} // namespace des


#endif /* DES_NETLISTPARSER_H_ */
