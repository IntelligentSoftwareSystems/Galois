#ifndef DYNAMIC_LIB_H
#define DYNAMIC_LIB_H

#include <string>

class DLOpenError : public std::exception 
{
  private:
      char *reason;
  public:
      DLOpenError(char *what) : reason(what) {};
      virtual const char* what() const throw()
      {
          return reason;
      }
      virtual ~DLOpenError() throw() {
          delete reason;
      }
};
  

class DynamicLib
{

  private:
    std::string path;
    void *handle = NULL; 

  public:
    // you can either pass handle or path for library.
    // Passing path leads to loading library and fullfill handle
    DynamicLib(const std::string &path);
    DynamicLib(const void *handle);

    ~DynamicLib();

    // load must be called before resolving symbols
    void load();
    void *resolvSymbol(const std::string &symbol);

};

#endif
