#ifndef __SCSIM_DEBUG_H
#define __SCSIM_DEBUG_H

#include <stdio.h>

#include "terminal-colors.h"

// debug categories
typedef enum debug_category {
    DEBUG_MethodFlag::UNPROTECTED = 0, //
    DEBUG_MUTEX = ( 1 << 1 ),
    DEBUG_LIFEDEATH = ( 1 << 2 ),
    DEBUG_CONDVAR = ( 1 << 3 ),
    DEBUG_JOIN = ( 1 << 4 ),
    DEBUG_BARRIER = ( 1 << 5 ),
    DEBUG_ALL = 0xFFFFFFFF
} debug_category_t;
// NB: the above is not 64-bit clean!!

/** the category of debugging currently enabled */
extern int DEBUG_CATEGORY;

static inline void NO_DEBUG( void ) {
    DEBUG_CATEGORY = DEBUG_MethodFlag::UNPROTECTED;
}
static inline void ALL_DEBUG( void ) {
    DEBUG_CATEGORY = DEBUG_ALL;
}
/** add debugging for a specified category */
static inline void DEBUG_CAT( debug_category_t category ) {
    DEBUG_CATEGORY = ( DEBUG_CATEGORY | category );
}
/** remove debugging for a specified category */
static inline void UNDEBUG_CAT( debug_category_t category ) {
    DEBUG_CATEGORY = ( DEBUG_CATEGORY & ~category );
}
/** check if any category in categories is being debugged */
static inline unsigned IS_DEBUGGED( debug_category_t categories ) {
    return ( ( DEBUG_CATEGORY & categories ) != 0 );
}

static inline void printColor( debug_category_t categories ) {
    if ( categories & DEBUG_MUTEX ) {
        fprintf( stdout, GREEN);
    } else if ( categories & DEBUG_LIFEDEATH ) {
        fprintf( stdout, CYAN);
    } else if ( categories & DEBUG_CONDVAR ) {
        fprintf( stdout, WHITE);
    } else if ( categories & DEBUG_JOIN ) {
        fprintf( stdout, BLUE);
    } else if ( categories & DEBUG_BARRIER ) {
        fprintf( stdout, WHITE);
    } else fprintf( stdout, " " );
}

/**
 * print out the specified message if debugging is enabled for any of
 * the specified categories
 * @param CATEGORIES a bitfield representing the categories associated
 * with this message
 * @param FORMAT... arguments to printf()
 */
#define DEBUG_MSG(CATEGORIES, FORMAT, ...) { \
    if ( IS_DEBUGGED( CATEGORIES ) ) { printColor(CATEGORIES); \
        fprintf( stdout, FORMAT CLEAR "\n", ##__VA_ARGS__ ); \
        fflush( stdout ); \
    } \
}

#include <stdlib.h> // for abort()
#undef assert
#define assert(EXPR) ASSERT(EXPR)
#undef ASSERT
/** fancy version of C's assert() that prints source code context and
 the expression that caused the assert to fail. Does not use
 assert() internally; uses abort() instead. */
#define ASSERT(EXPR) {\
  if ( !(EXPR) ) {\
    fprintf( stdout, ERROR "FAILED ASSERTION" CLEAR ": '" #EXPR "' at %s() in %s:%d\n",\
             __PRETTY_FUNCTION__, __FILE__, __LINE__ );\
    fflush( stdout );\
    abort();\
  }\
}
/** assert() with a message attached */
#define assert_msg(EXPR, MESSAGE) ASSERT_MSG(EXPR, MESSAGE)
#undef ASSERT_MSG
#define ASSERT_MSG(EXPR, MESSAGE) {\
  if ( !(EXPR) ) {\
    fprintf( stdout, ERROR "FAILED ASSERTION: %s" CLEAR " '" #EXPR "' at %s() in %s:%d.\n",\
             MESSAGE, __PRETTY_FUNCTION__, __FILE__, __LINE__ );\
    fflush( stdout );\
    abort();\
  }\
}

/** error message */
#define ERROR_MSG(MESSAGE) {\
  fprintf( stdout, ERROR "ERROR: %s " CLEAR " at %s() in %s:%d.\n",\
           MESSAGE, __PRETTY_FUNCTION__, __FILE__, __LINE__ );\
  fflush( stdout );\
  abort();\
}
/** error expression + message. NB: EXPR must be of type int */
#define ERROR_XMSG(EXPR, MESSAGE) {\
  fprintf( stdout, ERROR "ERROR: %s" CLEAR " at %s() in %s:%d. Expr '%s' has value: '%d'.\n",\
           MESSAGE, __PRETTY_FUNCTION__, __FILE__, __LINE__, #EXPR, (EXPR) );\
  fflush( stdout );\
  abort();\
}

/** warning message */
#define WARNING_MSG(MESSAGE) {\
  fprintf( stdout, WARNING "WARNING: %s" CLEAR " at %s() in %s:%d.\n",\
           MESSAGE, __PRETTY_FUNCTION__, __FILE__, __LINE__ );\
  fflush( stdout );\
}
/** warning expression + message. NB: EXPR must be of type int */
#define WARNING_XMSG(EXPR, MESSAGE) {\
  fprintf( stdout, WARNING "WARNING: %s" CLEAR " at %s() in %s:%d. Expr '%s' has value: '%d'.\n",\
           MESSAGE, __PRETTY_FUNCTION__, __FILE__, __LINE__, #EXPR, (EXPR) );\
  fflush( stdout );\
}

/** like a warning, but less emphatic, and sans source context */
static inline void ADVISORY_MSG( const char* message ) {
    fprintf( stdout, ADVISORY "NB: %s\n" CLEAR, message );
    fflush( stdout );
}

/**
 * print out an info message - no source code context info, and only
 * goes to stdout
 * @param FORMAT... arguments to printf()
 */
#define INFO_MSG(FORMAT, ...) {					\
   fprintf( stdout, FORMAT CLEAR "\n", ##__VA_ARGS__ );		\
   fflush( stdout );						\
 }

#endif
