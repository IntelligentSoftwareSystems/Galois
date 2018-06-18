/*
 * utilstd::string.h
 *
 *  Created on: 22/07/2014
 *      Author: jodymaick
 */

#ifndef UTILSTRING_H_
#define UTILSTRING_H_

#include <string>
#include <vector>

/**
 * Splits the given std::string in a vector of std::strings at each occurrence
 * of a given delimiter.
 *
 * @param str The std::string to be splitted.
 * @param delim The delimiter.
 * @param parts The vector comprising the parts of the split std::string.
 */
void split(const std::string& str, const std::string& delim,
           std::vector<std::string>& parts);

/**
 * Splits the given std::string at each occurrence of a given regex-based
 * delimiter and returns a vector of std::strings.
 *
 * @param s The std::string to be splitted.
 * @param rgx_str The regex-based delimiter.
 * @return The vector comprising the parts of the split std::string.
 */
// std::vector<std::string> regex_split(const std::string & s, std::string
// rgx_str = "\\s+");

/**
 * Checks if a given std::string starts with another given std::string.
 *
 * @param str The std::string to be checked in.
 * @param part The part to be searched into the std::string.
 * @return True if str starts with part. False otherwise.
 */
bool startsWith(std::string str, std::string part);

/**
 * Checks if a given std::string ends with another given std::string.
 *
 * @param str The std::string to be checked in.
 * @param part The part to be searched into the std::string.
 * @return True if str ends with part. False otherwise.
 */
bool endsWith(std::string str, std::string part);

/**
 * Returns a formatted std::string (just like it would be printed with printf).
 *
 * @param fmt The desired format of the std::string.
 * @return The formatted std::string.
 */
std::string format(const std::string fmt, ...);

void find_and_replace(std::string& source, std::string const& find,
                      std::string const& replace);

std::string get_clean_string(std::string string);

#endif /* UTILstd::string_H_ */
