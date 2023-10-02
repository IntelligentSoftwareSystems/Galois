//===------------------------------------------------------------*- C++ -*-===//
//
//                                     SHAD
//
//      The Scalable High-performance Algorithms and Data Structure Library
//
//===----------------------------------------------------------------------===//
//
// Copyright 2018 Battelle Memorial Institute
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
//===----------------------------------------------------------------------===//

#ifndef LIBGALOIS_INCLUDE_SHAD_DATATYPES_H_
#define LIBGALOIS_INCLUDE_SHAD_DATATYPES_H_

#include <ctime>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace shad {

/// @brief Data conversion utilities.
///
/// Please refer to methods specialization to check
/// which data types are supported.
namespace data_types {

/// @brief Enumeration of supported data types.
///
/// The enumeration is meant to be used when parsing data
/// (i.e. type information is not known at compile time).
enum data_t {
  STRING = 0,  // string support is currenlty limited
  CHARS,       // sequence of characters
  UINT,        // unsigned, binds by default to uint64_t
  INT,         // int, binds by default to int64_t
  FLOAT,       // float, binds by default to float
  DOUBLE,      // double, binds by default to double
  BOOL,        // bool, binds by default to bool
  DATE,        // date in "%y-%m-%d" format, binds by default to time_t
  USDATE,      // date in "%m/%d/%y" format, binds by default to time_t
  DATE_TIME,   // date in "%y-%m-%dT%H:%M:%S" format,
               // binds by default to time_t
  IP_ADDRESS,  // IPv4, binds by default to data_types::ipv4_t
  LIST_UINT,   // Sequence of unsigneds, support currently limited
  LIST_INT,    // Sequence of integers, support currently limited
  LIST_DOUBLE, // Sequence of doubles, support currently limited
  NONE
};

/// @brief Data structures for storing schema information.
/// Given a tuple of data, it associates elements labels and data types
/// to their position in the tuple.
using schema_t = std::vector<std::pair<std::string, data_t>>;

/// @brief Encoded null value.
/// @tparam ENC_t encoding type.
/// @return Encoded null value for ENC_t.
template <typename ENC_t>
constexpr ENC_t kNullValue = ENC_t();

/// @brief Encoded null value for uint64_t.
/// @return Null encoded value for uint64_t.
template <>
constexpr uint64_t kNullValue<uint64_t> = std::numeric_limits<int64_t>::max();

/// @brief Encoded null value for time_t (same as long).
/// @return Null encoded value for time_t (same as long).
template <>
constexpr time_t kNullValue<time_t> = std::numeric_limits<time_t>::max();

/// @brief Encoded null value for double.
/// @return Null encoded value for double.
template <>
constexpr double kNullValue<double> = std::numeric_limits<double>::max();

/// @brief Encode Function
/// Available specializations:
///    ENC_t = uint64_t, IN_t = std::string
/// @tparam ENC_t The type to encode to.
/// @tparam IN_t The type (format) of the data to encode.
/// @tparam DT data_types::data_t of the data to encode.
/// @param in Data to encode.
/// @return Encoded data.
template <typename ENC_t, typename IN_t, data_t DT>
ENC_t encode(IN_t& in);

/// @brief Encode Function
/// Available specializations:
///    ENC_t = uint64_t, IN_t = default bindings of data_types::data_t
/// @tparam ENC_t The type to encode to.
/// @tparam IN_t The type of the data to encode.
/// @param in Data to encode.
/// @return Encoded data.
template <typename ENC_t, typename IN_t>
ENC_t encode(IN_t& in);

template <typename ENC_t, typename IN_t>
ENC_t encode(IN_t& in, data_t dt);

template <typename ENC_t, size_t MAX_s, data_t ST>
std::array<ENC_t, MAX_s> encode(std::string& str) {
  std::array<ENC_t, MAX_s> res;
  if (str.size() > 0) {
    memcpy(res.data(), str.data(), sizeof(ENC_t) * MAX_s);
  } else {
    res.fill('\0');
  }
  return res;
}

template <typename ENC_t, typename DEC_t>
typename std::enable_if<(std::is_arithmetic<DEC_t>::value or
                         (sizeof(DEC_t) == sizeof(ENC_t))),
                        DEC_t>::type
decode(ENC_t encvalue) {
  DEC_t val;
  memcpy(&val, &encvalue, sizeof(DEC_t));
  return val;
}

template <typename ENC_t, typename DEC_t, data_t ST>
DEC_t decode(ENC_t value);

template <typename ENC_t, data_t ST>
typename std::enable_if<(ST == data_t::INT), int64_t>::type
decode(ENC_t encvalue) {
  return decode<ENC_t, int64_t>(encvalue);
}

template <typename ENC_t, data_t ST>
typename std::enable_if<(ST == data_t::UINT), uint64_t>::type
decode(ENC_t encvalue) {
  return decode<ENC_t, uint64_t>(encvalue);
}

template <typename ENC_t, data_t ST>
typename std::enable_if<(ST == data_t::FLOAT), float>::type
decode(ENC_t encvalue) {
  return decode<ENC_t, float>(encvalue);
}

template <typename ENC_t, data_t ST>
typename std::enable_if<(ST == data_t::DOUBLE), double>::type
decode(ENC_t encvalue) {
  return decode<ENC_t, double>(encvalue);
}

template <typename ENC_t, data_t ST>
typename std::enable_if<(ST == data_t::BOOL), bool>::type
decode(ENC_t encvalue) {
  return decode<ENC_t, bool>(encvalue);
}

template <typename ENC_t, data_t ST>
typename std::enable_if<(ST == data_t::DATE), std::time_t>::type
decode(ENC_t encvalue) {
  return decode<ENC_t, std::time_t>(encvalue);
}

template <typename ENC_t, size_t MAX_s, data_t ST>
std::string decode(std::array<ENC_t, MAX_s>& val) {
  return std::string(reinterpret_cast<const char*>(val.data()));
}
} // namespace data_types

// ENCODE METHODS SPECIALIZATION FOR UINT64 ENC_t
template <>
inline uint64_t
data_types::encode<uint64_t, std::string, data_types::UINT>(std::string& str) {
  uint64_t value;
  try {
    value = std::stoull(str);
  } catch (...) {
    value = kNullValue<uint64_t>;
  }
  return value;
}

template <>
inline uint64_t
data_types::encode<uint64_t, std::string, data_types::INT>(std::string& str) {
  uint64_t encval;
  int64_t value;
  try {
    value = stoll(str);
  } catch (...) {
    return kNullValue<uint64_t>;
  }
  memcpy(&encval, &value, sizeof(value));
  return encval;
}

template <>
inline uint64_t
data_types::encode<uint64_t, std::string, data_types::FLOAT>(std::string& str) {
  uint64_t encval;
  float value;
  try {
    value = stof(str);
  } catch (...) {
    return kNullValue<uint64_t>;
  }
  memcpy(&encval, &value, sizeof(value));
  return encval;
}

template <>
inline uint64_t data_types::encode<uint64_t, std::string, data_types::DOUBLE>(
    std::string& str) {
  uint64_t encval;
  double value;
  try {
    value = stod(str);
  } catch (...) {
    return kNullValue<uint64_t>;
  }
  memcpy(&encval, &value, sizeof(value));
  return encval;
}

template <>
inline uint64_t
data_types::encode<uint64_t, std::string, data_types::BOOL>(std::string& str) {
  if (str.size() == 0)
    return kNullValue<uint64_t>;
  uint64_t encval = 1;
  if ((str == "F") || (str == "f") || (str == "FALSE") || (str == "false") ||
      (str == "0"))
    encval = 0;
  return encval;
}

template <>
inline uint64_t
data_types::encode<uint64_t, std::string, data_types::CHARS>(std::string& str) {
  uint64_t encval = 0;
  memset(&encval, '\0', sizeof(encval));
  memcpy(&encval, str.c_str(), sizeof(encval) - 1);
  return encval;
}

template <>
inline uint64_t
data_types::encode<uint64_t, std::string, data_types::IP_ADDRESS>(
    std::string& str) {
  uint64_t val, value = 0;
  std::string::iterator start = str.begin();
  for (unsigned i = 0; i < 4; i++) {
    std::string::iterator end = std::find(start, str.end(), '.');
    try {
      val = std::stoull(std::string(start, end));
    } catch (...) {
      return kNullValue<uint64_t>;
    }
    if (val < 256) {
      value = (value << 8) + val;
      start = end + 1;
    } else {
      return kNullValue<uint64_t>;
    }
  }
  return value;
}

template <>
inline uint64_t
data_types::encode<uint64_t, std::string, data_types::DATE>(std::string& str) {
  uint64_t value = 0;
  struct tm date {};
  date.tm_isdst = -1;
  strptime(str.c_str(), "%Y-%m-%d", &date);
  time_t t;
  try {
    t = mktime(&date);
  } catch (...) {
    return kNullValue<uint64_t>;
  }
  memcpy(&value, &t, sizeof(value));
  return value;
}

template <>
inline uint64_t data_types::encode<uint64_t, std::string, data_types::USDATE>(
    std::string& str) {
  uint64_t value = 0;
  struct tm date {};
  date.tm_isdst = -1;
  strptime(str.c_str(), "%m/%d/%y", &date);
  time_t t;
  try {
    t = mktime(&date);
  } catch (...) {
    return kNullValue<uint64_t>;
  }
  memcpy(&value, &t, sizeof(value));
  return value;
}

template <>
inline uint64_t
data_types::encode<uint64_t, std::string, data_types::DATE_TIME>(
    std::string& str) {
  uint64_t value = 0;
  struct tm date {};
  date.tm_isdst = -1;
  strptime(str.c_str(), "%Y-%m-%dT%H:%M:%S", &date);
  time_t t;
  try {
    t = mktime(&date);
  } catch (...) {
    return kNullValue<uint64_t>;
  }
  memcpy(&value, &t, sizeof(value));
  return value;
}

// ENCODE METHODS SPECIALIZATION FOR DOUBLE ENC_t

template <>
inline double
data_types::encode<double, std::string, data_types::UINT>(std::string& str) {
  double encval;
  uint64_t value;
  try {
    value = std::stoull(str);
  } catch (...) {
    return kNullValue<double>;
  }
  memcpy(&encval, &value, sizeof(value));
  return encval;
}

template <>
inline double
data_types::encode<double, std::string, data_types::INT>(std::string& str) {
  double encval;
  int64_t value;
  try {
    value = stoll(str);
  } catch (...) {
    return kNullValue<double>;
  }
  memcpy(&encval, &value, sizeof(value));
  return encval;
}

template <>
inline double
data_types::encode<double, std::string, data_types::FLOAT>(std::string& str) {
  double encval;
  float value;
  try {
    value = stof(str);
  } catch (...) {
    return kNullValue<double>;
  }
  memcpy(&encval, &value, sizeof(value));
  return encval;
}

template <>
inline double
data_types::encode<double, std::string, data_types::DOUBLE>(std::string& str) {
  double value;
  try {
    value = stod(str);
  } catch (...) {
    return kNullValue<double>;
  }
  return value;
}

template <>
inline double
data_types::encode<double, std::string, data_types::BOOL>(std::string& str) {
  if (str.size() == 0)
    return kNullValue<uint64_t>;
  double encval = 1;
  if ((str == "F") || (str == "f") || (str == "FALSE") || (str == "false") ||
      (str == "0"))
    encval = 0;
  return encval;
}

template <>
inline double
data_types::encode<double, std::string, data_types::CHARS>(std::string& str) {
  double encval = 0;
  memset(&encval, '\0', sizeof(encval));
  memcpy(&encval, str.c_str(), sizeof(encval) - 1);
  return encval;
}

template <>
inline double data_types::encode<double, std::string, data_types::IP_ADDRESS>(
    std::string& str) {
  uint64_t val, value = 0;
  std::string::iterator start = str.begin();
  for (unsigned i = 0; i < 4; i++) {
    std::string::iterator end = std::find(start, str.end(), '.');
    try {
      val = std::stoull(std::string(start, end));
    } catch (...) {
      return kNullValue<double>;
    }
    if (val < 256) {
      value = (value << 8) + val;
      start = end + 1;
    } else {
      return kNullValue<double>;
    }
  }
  double encval;
  memcpy(&encval, &value, sizeof(value));
  return encval;
}

template <>
inline double
data_types::encode<double, std::string, data_types::DATE>(std::string& str) {
  double value = 0;
  struct tm date {};
  date.tm_isdst = -1;
  strptime(str.c_str(), "%Y-%m-%d", &date);
  time_t t;
  try {
    t = mktime(&date);
  } catch (...) {
    return kNullValue<double>;
  }
  memcpy(&value, &t, sizeof(value));
  return value;
}

template <>
inline double
data_types::encode<double, std::string, data_types::USDATE>(std::string& str) {
  double value = 0;
  struct tm date {};
  date.tm_isdst = -1;
  strptime(str.c_str(), "%m/%d/%y", &date);
  time_t t;
  try {
    t = mktime(&date);
  } catch (...) {
    return kNullValue<uint64_t>;
  }
  memcpy(&value, &t, sizeof(value));
  return value;
}

template <>
inline double data_types::encode<double, std::string, data_types::DATE_TIME>(
    std::string& str) {
  double value = 0;
  struct tm date {};
  date.tm_isdst = -1;
  strptime(str.c_str(), "%Y-%m-%dT%H:%M:%S", &date);
  time_t t;
  try {
    t = mktime(&date);
  } catch (...) {
    return kNullValue<uint64_t>;
  }
  memcpy(&value, &t, sizeof(value));
  return value;
}

// ENCODE METHODS SPECIALIZATION FOR TIME_T ENC_t (same as long)
template <>
inline time_t
data_types::encode<time_t, std::string, data_types::UINT>(std::string& str) {
  time_t value;
  try {
    value = std::stoul(str);
  } catch (...) {
    value = kNullValue<time_t>;
  }
  return value;
}

template <>
inline time_t
data_types::encode<time_t, std::string, data_types::INT>(std::string& str) {
  int64_t value;
  try {
    value = stol(str);
  } catch (...) {
    return kNullValue<time_t>;
  }
  return value;
}

template <>
inline time_t
data_types::encode<time_t, std::string, data_types::FLOAT>(std::string& str) {
  time_t encval;
  float value;
  try {
    value = stof(str);
  } catch (...) {
    return kNullValue<time_t>;
  }
  memcpy(&encval, &value, sizeof(value));
  return encval;
}

template <>
inline time_t
data_types::encode<time_t, std::string, data_types::DOUBLE>(std::string& str) {
  time_t encval;
  double value;
  try {
    value = stod(str);
  } catch (...) {
    return kNullValue<time_t>;
  }
  memcpy(&encval, &value, sizeof(value));
  return encval;
}

template <>
inline time_t
data_types::encode<time_t, std::string, data_types::BOOL>(std::string& str) {
  if (str.size() == 0)
    return kNullValue<uint64_t>;
  time_t encval = 1;
  if ((str == "F") || (str == "f") || (str == "FALSE") || (str == "false") ||
      (str == "0"))
    encval = 0;
  return encval;
}

template <>
inline time_t
data_types::encode<time_t, std::string, data_types::CHARS>(std::string& str) {
  time_t encval = 0;
  memset(&encval, '\0', sizeof(encval));
  memcpy(&encval, str.c_str(), sizeof(encval) - 1);
  return encval;
}

template <>
inline time_t data_types::encode<time_t, std::string, data_types::IP_ADDRESS>(
    std::string& str) {
  time_t val, value = 0;
  std::string::iterator start = str.begin();
  for (unsigned i = 0; i < 4; i++) {
    std::string::iterator end = std::find(start, str.end(), '.');
    try {
      val = std::stoull(std::string(start, end));
    } catch (...) {
      return kNullValue<time_t>;
    }
    if (val < 256) {
      value = (value << 8) + val;
      start = end + 1;
    } else {
      return kNullValue<time_t>;
    }
  }
  return value;
}

template <>
inline time_t
data_types::encode<time_t, std::string, data_types::DATE>(std::string& str) {
  struct tm date {};
  date.tm_isdst = -1;
  strptime(str.c_str(), "%Y-%m-%d", &date);
  time_t t;
  try {
    t = mktime(&date);
  } catch (...) {
    return kNullValue<time_t>;
  }
  return t;
}

template <>
inline time_t
data_types::encode<time_t, std::string, data_types::USDATE>(std::string& str) {
  struct tm date {};
  date.tm_isdst = -1;
  strptime(str.c_str(), "%m/%d/%y", &date);
  time_t t;
  try {
    t = mktime(&date);
  } catch (...) {
    return kNullValue<time_t>;
  }
  return t;
}

template <>
inline time_t data_types::encode<time_t, std::string, data_types::DATE_TIME>(
    std::string& str) {
  struct tm date {};
  date.tm_isdst = -1;
  strptime(str.c_str(), "%Y-%m-%dT%H:%M:%S", &date);
  time_t t;
  try {
    t = mktime(&date);
  } catch (...) {
    return kNullValue<uint64_t>;
  }
  return t;
}

template <typename ENC_t, typename IN_t>
ENC_t data_types::encode(IN_t& in, data_types::data_t dt) {
  switch (dt) {
    //     case data_types::STRING :
    //       return data_types::encode<ENC_t, IN_t, data_types::STRING>(in);
    //     case data_types::CHARS :
    //       return data_types::encode<ENC_t, IN_t, data_types::CHARS>(in);
  case data_types::UINT:
    return data_types::encode<ENC_t, IN_t, data_types::UINT>(in);
  case data_types::INT:
    return data_types::encode<ENC_t, IN_t, data_types::INT>(in);
  case data_types::FLOAT:
    return data_types::encode<ENC_t, IN_t, data_types::FLOAT>(in);
  case data_types::DOUBLE:
    return data_types::encode<ENC_t, IN_t, data_types::DOUBLE>(in);
  case data_types::BOOL:
    return data_types::encode<ENC_t, IN_t, data_types::BOOL>(in);
  case data_types::DATE:
    return data_types::encode<ENC_t, IN_t, data_types::DATE>(in);
  case data_types::USDATE:
    return data_types::encode<ENC_t, IN_t, data_types::USDATE>(in);
  case data_types::DATE_TIME:
    return data_types::encode<ENC_t, IN_t, data_types::DATE_TIME>(in);
  case data_types::IP_ADDRESS:
    return data_types::encode<ENC_t, IN_t, data_types::IP_ADDRESS>(in);
  }
  return data_types::kNullValue<ENC_t>;
}

template <>
inline std::string
data_types::decode<uint64_t, std::string, data_types::UINT>(uint64_t value) {
  if (value == kNullValue<uint64_t>)
    return "";
  return std::to_string(value);
}

template <>
inline std::string
data_types::decode<uint64_t, std::string, data_types::INT>(uint64_t value) {
  if (value == kNullValue<uint64_t>)
    return "";
  int64_t v;
  memcpy(&v, &value, sizeof(v));
  return std::to_string(v);
}

template <>
inline std::string
data_types::decode<uint64_t, std::string, data_types::FLOAT>(uint64_t value) {
  if (value == kNullValue<uint64_t>)
    return "";
  float v;
  memcpy(&v, &value, sizeof(v));
  return std::to_string(v);
}

template <>
inline std::string
data_types::decode<uint64_t, std::string, data_types::DOUBLE>(uint64_t value) {
  if (value == kNullValue<uint64_t>)
    return "";
  double v;
  memcpy(&v, &value, sizeof(v));
  return std::to_string(v);
}

template <>
inline std::string
data_types::decode<uint64_t, std::string, data_types::IP_ADDRESS>(
    uint64_t value) {
  std::string ipAddr = "";
  uint64_t octets[4];
  for (uint64_t k = 0; k < 4; k++) {
    octets[k] = value & 255;
    value     = value >> 8;
  }
  for (uint64_t k = 3; k >= 1; k--)
    ipAddr += std::to_string(octets[k]) + '.';
  return ipAddr + std::to_string(octets[0]);
}

template <>
inline std::string
data_types::decode<uint64_t, std::string, data_types::BOOL>(uint64_t value) {
  if (value == kNullValue<uint64_t>)
    return "";
  return std::to_string(value);
}

template <>
inline std::string
data_types::decode<uint64_t, std::string, data_types::DATE>(uint64_t value) {
  time_t t = data_types::decode<uint64_t, data_types::DATE>(value);
  char dateString[11];
  strftime(dateString, 11, "%Y-%m-%d", std::localtime(&t));
  return std::string(dateString);
}

template <>
inline std::string
data_types::decode<uint64_t, std::string, data_types::CHARS>(uint64_t value) {
  const char* c = reinterpret_cast<const char*>(&value);
  return std::string(c);
}

template <>
inline uint64_t data_types::decode<uint64_t, uint64_t>(uint64_t encvalue) {
  return encvalue;
}
} // namespace shad

#endif // LIBGALOIS_INCLUDE_SHAD_DATA_TYPES_H_
