#ifndef DES_LOGIC_DEFS_H_
#define DES_LOGIC_DEFS_H_


namespace des {

/** type used for value of a signal e.g. 0, 1, X , Z */
typedef char LogicVal;

/** the unknown logic value */
const char LOGIC_UNKNOWN = 'X';
const char LOGIC_ZERO = '0';
const char LOGIC_ONE = '1';


} // namespace des

#endif
