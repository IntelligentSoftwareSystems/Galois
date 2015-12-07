#ifndef ELEMENT_HPP
#define ELEMENT_HPP

#include <vector>

class Element {
    public:
        uint64_t x1, y1;
        uint64_t x2, y2;
        uint64_t k, l;
        std::vector<uint64_t> dofs;
};

#endif // ELEMENT_HPP
