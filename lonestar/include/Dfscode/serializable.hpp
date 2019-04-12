#ifndef __SERIALIZABLE_HPP__
#define __SERIALIZABLE_HPP__

class serializable_buffer {
public:
  virtual size_t get_serialized_size() const = 0;
  virtual size_t get_serialized_size(char *buffer, size_t buffer_size) const = 0;
  virtual size_t serialize(char *buffer, size_t buffer_size) const = 0;
  virtual size_t deserialize(char *buffer, size_t buffer_size) = 0;
};

class serializable_stream {
public:
  virtual size_t serialize(std::ostream &) const = 0;
  virtual size_t deserialize(std::istream &) = 0;
};

class serializable : public serializable_buffer, public serializable_stream {
public:
  virtual ~serializable() {
  }
};

#endif

