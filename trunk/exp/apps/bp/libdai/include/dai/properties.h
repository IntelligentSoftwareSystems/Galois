/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
 */


/// \file
/// \brief Defines the Property and PropertySet classes, which are mainly used for managing parameters of inference algorithms


#ifndef __defined_libdai_properties_h
#define __defined_libdai_properties_h


#include <iostream>
#include <sstream>
#include <boost/any.hpp>
#include <map>
#include <vector>
#include <typeinfo>
#include <dai/exceptions.h>
#include <dai/util.h>
#include <boost/lexical_cast.hpp>


namespace dai {


/// Type of the key of a Property
typedef std::string PropertyKey;

/// Type of the value of a Property
typedef boost::any  PropertyValue;

/// A Property is a pair of a key and a corresponding value
typedef std::pair<PropertyKey, PropertyValue> Property;


/// Writes a Property object (key-value pair) to an output stream
/** \note Not all value types are automatically supported; if a type is unknown, an 
 *  UNKNOWN_PROPERTY_TYPE exception is thrown. Adding support for a new type can 
 *  be done by changing this function body.
 */
std::ostream& operator<< ( std::ostream & os, const Property &p );


/// Represents a set of properties, mapping keys (of type PropertyKey) to values (of type PropertyValue)
/** Properties are used for specifying parameters of algorithms in a convenient way, where the values of
 *  the parameters can be of different types (e.g., strings, doubles, integers, enums). A PropertySet is
 *  an attempt to mimic the functionality of a Python dictionary object in C++, using the boost::any class. 
 *
 *  A PropertySet can be converted to and from a string, using the following format:
 *
 *  <tt>[key1=val1,key2=val2,...,keyn=valn]</tt>
 *
 *  That is,
 *  - the whole PropertySet is wrapped in square brackets ("[", "]")
 *  - all properties in the PropertySet are seperated by a comma (",")
 *  - each Property consists of:
 *    - the name of the key
 *    - an equality sign ("=")
 *    - its value (represented as a string)
 *
 *  Also, a PropertySet provides functionality for converting the representation of
 *  individual values from some arbitrary type to and from std::string.
 *
 *  \note Not all types are automatically supported; if a type is unknown, an UNKNOWN_PROPERTY_TYPE 
 *  exception is thrown. Adding support for a new type can be done in the body of the
 *  operator<<(std::ostream &, const Property &).
 */
class PropertySet : private std::map<PropertyKey, PropertyValue> {
    public:
    /// \name Constructors and destructors
    //@{
        /// Default constructor
        PropertySet() {}

        /// Construct from a string
        /** \param s string in the format <tt>"[key1=val1,key2=val2,...,keyn=valn]"</tt>
         */
        PropertySet( const std::string& s ) {
            std::stringstream ss;
            ss << s;
            ss >> *this;
        }
    //@}

    /// \name Setting property keys/values
    //@{
        /// Sets a property (a key \a key with a corresponding value \a val)
        PropertySet& set( const PropertyKey& key, const PropertyValue& val ) { 
            this->operator[](key) = val; 
            return *this; 
        }

        /// Set properties according to \a newProps, overriding properties that already exist with new values
        PropertySet& set( const PropertySet& newProps ) {
            const std::map<PropertyKey, PropertyValue> *m = &newProps;
            foreach(value_type i, *m)
                set( i.first, i.second );
            return *this;
        }

        /// Shorthand for (temporarily) adding properties
        /** \par Example:
            \code
PropertySet p()("method","BP")("verbose",1)("tol",1e-9)
            \endcode
         */
        PropertySet operator()( const PropertyKey& key, const PropertyValue& val ) const { 
            PropertySet copy = *this; 
            return copy.set(key,val); 
        }

        /// Sets a property (a key \a key with a corresponding value \a val, which is first converted from \a ValueType to string)
        /** The implementation makes use of boost::lexical_cast.
         *  \tparam ValueType Type from which the value should be cast
         *  \throw IMPOSSIBLE_TYPECAST if the type cast cannot be done
         */
        template<typename ValueType>
        PropertySet& setAsString( const PropertyKey& key, const ValueType& val ) {
            try {
                return set( key, boost::lexical_cast<std::string>(val) );
            } catch( boost::bad_lexical_cast & ) {
                DAI_THROWE(IMPOSSIBLE_TYPECAST,"Cannot cast value of property '" + key + "' to string.");
            }
        }

        /// Converts the type of the property value corresponding with \a key from string to \a ValueType (if necessary)
        /** The implementation makes use of boost::lexical_cast
         *  \tparam ValueType Type to which the value should be cast
         *  \throw IMPOSSIBLE_TYPECAST if the type cast cannot be done
         */
        template<typename ValueType>
        void convertTo( const PropertyKey& key ) { 
            PropertyValue val = get(key);
            if( val.type() != typeid(ValueType) ) {
                DAI_ASSERT( val.type() == typeid(std::string) );
                try {
                    set(key, boost::lexical_cast<ValueType>(getAs<std::string>(key)));
                } catch(boost::bad_lexical_cast &) {
                    DAI_THROWE(IMPOSSIBLE_TYPECAST,"Cannot cast value of property '" + key + "' from string to desired type.");
                }
            }
        }
    //@}

    //@}

    /// \name Queries
    //@{
        /// Return number of key-value pairs
        size_t size() const {
            return std::map<PropertyKey, PropertyValue>::size();
        }

        /// Removes all key-value pairs
        void clear() {
            std::map<PropertyKey, PropertyValue>::clear();
        }

        /// Removes key-value pair with given \a key
        size_t erase( const PropertyKey &key ) {
            return std::map<PropertyKey, PropertyValue>::erase( key );
        }

        /// Check if a property with the given \a key is defined
        bool hasKey( const PropertyKey& key ) const { 
            PropertySet::const_iterator x = find(key); 
            return (x != this->end()); 
        }

        /// Returns a set containing all keys
        std::set<PropertyKey> keys() const {
            std::set<PropertyKey> res;
            const_iterator i;
            for( i = begin(); i != end(); i++ )
                res.insert( i->first );
            return res;
        }

        /// Gets the value corresponding to \a key
        /** \throw OBJECT_NOT_FOUND if the key cannot be found in \c *this
         */
        const PropertyValue& get( const PropertyKey& key ) const {
            PropertySet::const_iterator x = find(key);
            if( x == this->end() )
                DAI_THROWE(OBJECT_NOT_FOUND,"PropertySet::get cannot find property '" + key + "'");
            return x->second;
        }

        /// Gets the value corresponding to \a key, cast to \a ValueType
        /** \tparam ValueType Type to which the value should be cast
         *  \throw OBJECT_NOT_FOUND if the key cannot be found in \c *this
         *  \throw IMPOSSIBLE_TYPECAST if the type cast cannot be done
         */
        template<typename ValueType>
        ValueType getAs( const PropertyKey& key ) const {
            try {
                return boost::any_cast<ValueType>(get(key));
            } catch( const boost::bad_any_cast & ) {
                DAI_THROWE(IMPOSSIBLE_TYPECAST,"Cannot cast value of property '" + key + "' to desired type.");
                return ValueType();
            }
        }

        /// Gets the value corresponding to \a key, cast to \a ValueType, converting from a string if necessary
        /** If the type of the value is already equal to \a ValueType, no conversion is done.
         *  Otherwise, the type of the value should be a std::string, in which case boost::lexical_cast is
         *  used to convert this to \a ValueType.
         *  \tparam ValueType Type to which the value should be cast/converted
         *  \throw OBJECT_NOT_FOUND if the key cannot be found in \c *this
         *  \throw IMPOSSIBLE_TYPECAST if the type cast cannot be done
         */
        template<typename ValueType>
        ValueType getStringAs( const PropertyKey& key ) const { 
            PropertyValue val = get(key);
            if( val.type() == typeid(ValueType) ) {
                return boost::any_cast<ValueType>(val);
            } else if( val.type() == typeid(std::string) ) {
                try {
                    return boost::lexical_cast<ValueType>(getAs<std::string>(key));
                } catch(boost::bad_lexical_cast &) {
                    DAI_THROWE(IMPOSSIBLE_TYPECAST,"Cannot cast value of property '" + key + "' from string to desired type.");
                }
            } else
                DAI_THROWE(IMPOSSIBLE_TYPECAST,"Cannot cast value of property '" + key + "' from string to desired type.");
            return ValueType();
        }
    //@}

    /// \name Input/output
    //@{
        /// Writes a PropertySet object to an output stream.
        /** It uses the format <tt>"[key1=val1,key2=val2,...,keyn=valn]"</tt>.
         *  \note Only a subset of all possible types is supported (see the implementation of this function).
         *  Adding support for more types has to be done by hand.
         *  \throw UNKNOWN_PROPERTY_TYPE if the type of a property value is not supported.
         */
        friend std::ostream& operator<< ( std::ostream& os, const PropertySet& ps );

        /// Reads a PropertySet object from an input stream.
        /** It expects a string in the format <tt>"[key1=val1,key2=val2,...,keyn=valn]"</tt>.
         *  Values are stored as strings.
         *  \throw MALFORMED_PROPERTY if the string is not in the expected format
         */
        friend std::istream& operator>> ( std::istream& is, PropertySet& ps );
    //@}
};


} // end of namespace dai


#endif
