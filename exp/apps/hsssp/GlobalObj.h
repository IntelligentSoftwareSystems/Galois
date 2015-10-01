#include <vector>
#include <cstdint>

class GlobalObject {
  //FIXME: lock?
  static std::vector<uintptr_t> allobjs;
  uint32_t objID;

 protected:
  static uintptr_t ptrForObj(unsigned oid);

  GlobalObject(const GlobalObject&) = delete;
  GlobalObject(GlobalObject&&) = delete;

  template<typename T>
  GlobalObject(const T* ptr) {
    objID = allobjs.size();
    allobjs.push_back(reinterpret_cast<uintptr_t>(ptr));
  }

  uint32_t idForSelf() const {
    return objID;
  }
};
