#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#ifndef GALOIS_FILEREADER_H
#define GALOIS_FILEREADER_H

class FileReader {
  char *buffer, *bufferEnd, *cursor;
  size_t fileSize;
  char *delimiters;
  size_t numDelimiters;
  char *separators;
  size_t numSeparators;

private:
  bool isSeparator(char c);
  bool isDelimiter(char c);

private:
  std::vector<std::string> tokenStack;

public:
  FileReader(std::string inName, 
    char *delimiters, size_t numDelimiters, 
    char *separators, size_t numSeparators);
  ~FileReader();
  std::string nextToken();
  void pushToken(std::string token);
};

#endif // GALOIS_FILEREADER_H

