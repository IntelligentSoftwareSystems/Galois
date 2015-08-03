#ifndef ELEMENT_HPP
#define ELEMENT_HPP

#include <vector>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

class Element {
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
          ar & k & l;
          ar & supernodes;
        }        
    public:
        uint64_t k, l;
        std::vector<uint64_t> supernodes;
};

#endif // ELEMENT_HPP
