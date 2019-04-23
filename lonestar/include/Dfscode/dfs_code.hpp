#ifndef __DFS_CODE_HPP__
#define __DFS_CODE_HPP__
#include <string>
#include <vector>
#include <cstdio>
#include <sstream>
#include <string.h>
#include <iostream>
#include <stdexcept>
#include "types.hpp"

/*
using std::runtime_error;
class method_unimplemented : public std::runtime_error {
public:
  method_unimplemented(const char *err) : runtime_error(std::string("Method ") + err + "not implemented") {
  }
};

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
//*/
class DFS {
public:
	VeridT from;
	VeridT to;
	LabelT fromlabel;
	LabelT elabel;
	LabelT tolabel;
	friend bool operator==(const DFS &d1, const DFS &d2) {
		return (d1.from == d2.from && d1.to == d2.to && d1.fromlabel == d2.fromlabel
				&& d1.elabel == d2.elabel && d1.tolabel == d2.tolabel);
	}
	friend bool operator!=(const DFS &d1, const DFS &d2) {
		return (!(d1 == d2));
	}
	friend std::ostream &operator<<(std::ostream &out, const DFS &d) {
		out << d.to_string().c_str();
		return out;
	}
	friend bool operator<(const DFS &d1, const DFS &d2){
		if(d1.from < d2.from) return true;
		if(d1.from > d2.from) return false;
		if(d1.to < d2.to) return true;
		if(d1.to > d2.to) return false;
		if(d1.fromlabel < d2.fromlabel) return true;
		if(d1.fromlabel > d2.fromlabel) return false;
		if(d1.elabel < d2.elabel) return true;
		if(d1.elabel > d2.elabel) return false;
		if(d1.tolabel < d2.tolabel) return true;
		if(d1.tolabel > d2.tolabel) return false;
		return false;
	}

	DFS() : from(0), to(0), fromlabel(0), elabel(0), tolabel(0) {}
	DFS(VeridT from, VeridT to, LabelT fromlabel, LabelT elabel, LabelT tolabel) : from(from), to(to), fromlabel(fromlabel), elabel(elabel), tolabel(tolabel) {}
	DFS(char *buffer, int size);
	DFS(const DFS &d) : from(d.from), to(d.to), fromlabel(d.fromlabel), elabel(d.elabel), tolabel(d.tolabel) {}
	//std::string to_string(bool print_edge_type = true) const;
	std::string to_string(bool print_edge_type = true) const {
		std::stringstream ss;
		if(print_edge_type) {
			if(is_forward()) ss << "F";
			else ss << "B";
		}
		ss << "(" << from << " " << to << " " << fromlabel << " " << elabel << " " << tolabel << ")";
		return ss.str();
	}
	bool is_forward() const { return from < to; }
	bool is_backward() const { return from > to; }
/*
	//bool is_forward() const;
	//bool is_backward() const;
	static size_t deserialize(DFS &result, char *buf, size_t bufsize);
	static size_t serialize(DFS &result, char *buf, size_t bufsize);
	static size_t get_serialized_dfs_code_size(DFS &result);
	//static DFS parse_from_string(const char *str_dfs_code);
	static DFS parse_from_string(const char *str_dfs_code) {
		size_t len = strlen(str_dfs_code);
		int from;
		int to;
		int fromlabel;
		int elabel;
		int tolabel;
		char F_B;
		int r = sscanf(str_dfs_code, "%c(%d %d %d %d %d)", &F_B, &from, &to, &fromlabel, &elabel, &tolabel);
		if(r != 6) throw runtime_error("error: could not read dfs code from string");
		return DFS(from, to, fromlabel, elabel, tolabel);
	} // parse_from_string
	//virtual size_t get_serialized_size() const;
	//virtual size_t get_serialized_size(char *buffer, size_t buffer_size) const;
	//virtual size_t serialize(char *buffer, size_t buffer_size) const;
	//virtual size_t deserialize(char *buffer, size_t buffer_size);
	//virtual size_t serialize(std::ostream &) const;
	//virtual size_t deserialize(std::istream &);
	virtual size_t get_serialized_size() const {
		return sizeof(int) * 5;
	}
	virtual size_t get_serialized_size(char *buffer, size_t buffer_size) const {
		return *((int*)buffer);
	}
	virtual size_t serialize(char *buffer, size_t buffer_size) const {
		if(buffer_size < sizeof(int) * 5) throw runtime_error("Buffer too small");
		int *buf = (int*) buffer;
		buf[0] = from;
		buf[1] = to;
		buf[2] = fromlabel;
		buf[3] = elabel;
		buf[4] = tolabel;
		return 5 * sizeof(int);
	}
	virtual size_t deserialize(char *buffer, size_t buffer_size) {
		if(buffer_size < get_serialized_size()) throw runtime_error("Buffer too small");
		int *buf = (int*) buffer;
		from = buf[0];
		to = buf[1];
		fromlabel = buf[2];
		elabel = buf[3];
		tolabel = buf[4];
		return 5 * sizeof(int);
	}
	virtual size_t serialize(std::ostream &) const {
		throw method_unimplemented("DFS::serialize");
	}
	virtual size_t deserialize(std::istream &) {
		throw method_unimplemented("DFS::serialize");
	}
//*/
};

struct DFS_less_then {
	//bool bckwrd_bckwrd_less(const DFS &d1, const DFS &d2) const;
	//bool frwrd_bckwrd_less(const DFS &d1, const DFS &d2) const;
	//bool frwrd_frwrd_less(const DFS &d1, const DFS &d2) const;
	//bool operator()(const DFS &d1, const DFS &d2) const;
	bool bckwrd_bckwrd_less(const DFS &d1, const DFS &d2) const {
		return (d1.to < d2.to) || (d1.to == d2.to && d1.elabel < d2.elabel);
	}
	bool frwrd_bckwrd_less(const DFS &d1, const DFS &d2) const {
		if(d1.is_backward() && d2.is_forward()) return true;
		return false;
	}
	bool frwrd_frwrd_less(const DFS &d1, const DFS &d2) const {
		if(d1.from > d2.from) return true;
		if(d1.from < d2.from) return false;
		if(d1.from == d2.from) {
			//bool tmp = (d1.elabel < d2.elabel);
			if(d1.elabel < d2.elabel) return true;
			if(d1.elabel > d2.elabel) return false;
			return (d1.tolabel < d2.tolabel);
		} // if
		return false;
	}
	bool operator()(const DFS &d1, const DFS &d2) const {
		if(d1.is_backward() && d2.is_backward()) {
			bool result = bckwrd_bckwrd_less(d1, d2);
			return result;
		}
		if(d1.is_forward() && d2.is_forward()) {
			bool result = frwrd_frwrd_less(d1, d2);
			return result;
		}
		bool result = frwrd_bckwrd_less(d1, d2);
		return result;
	}
};

struct DFS_less_then_fast {
	bool operator()(const DFS &d1, const DFS &d2) const {
		if(d1.from < d2.from) return true;
		if(d1.from > d2.from) return false;
		if(d1.to < d2.to) return true;
		if(d1.to > d2.to) return false;
		if(d1.fromlabel < d2.fromlabel) return true;
		if(d1.fromlabel > d2.fromlabel) return false;
		if(d1.elabel < d2.elabel) return true;
		if(d1.elabel > d2.elabel) return false;
		if(d1.tolabel < d2.tolabel) return true;
		if(d1.tolabel > d2.tolabel) return false;
		return false;
	}
};

/**
 * Standard equal, as defined in DFS. The only difference is that it
 * is functor.
 */
struct DFS_std_equal {
	bool operator()(const DFS &d1, const DFS &d2) const {
		return d1 == d2;
	}
};

struct DFS_std_not_equal {
	bool operator()(const DFS &d1, const DFS &d2) const {
		return d1 != d2;
	}
};

/**
 * this is a special version of the == operator that compares the DFS
 * structure only partially, depending on whether it is forward or
 * backward edge.
 */
struct DFS_partial_equal {
	bool operator()(const DFS &d1, const DFS &d2) const {
		if(d1.from != d2.from || d1.to != d2.to) return false;
		if(d1.fromlabel != (LabelT)-1 && d2.fromlabel != (LabelT)-1 && d1.fromlabel != d2.fromlabel) return false;
		if(d1.tolabel != (LabelT)-1 && d2.tolabel != (LabelT)-1 && d1.tolabel != d2.tolabel) return false;
		if(d1.elabel != d2.elabel) return false;
		return true;
	} // operator()
};

/**
 * this is a special version of the == operator that compares the DFS
 * structure only partially, depending on whether it is forward or
 * backward edge. INTERNALLY USES DFS_equal.
 */
struct DFS_partial_not_equal {
	DFS_partial_equal eq;
	bool operator()(const DFS &d1, const DFS &d2) const {
		return !eq(d1, d2);
	}
};

class DFSCode;
std::ostream &operator<<(std::ostream &out, const DFSCode &code);

//struct DFSCode : public std::vector<DFS>, public serializable {
struct DFSCode : public std::vector<DFS> {
private:
	RMPath rmpath; // right-most path
public:
	const RMPath &get_rmpath() const { return rmpath; }
	// RMPath is in the opposite order then the DFS code, i.e., the
	// indexes into DFSCode go from higher numbers to lower numbers.
	const RMPath &buildRMPath() {
		rmpath.clear();
		VeridT old_from = (VeridT)-1;
		for(int i = size() - 1; i >= 0; --i) {
			if((*this)[i].from < (*this)[i].to &&  // forward
					(rmpath.empty() || old_from == (*this)[i].to)) {
				rmpath.push_back(i);
				old_from = (*this)[i].from;
			}
		}
		return rmpath;
	}
	// Convert current DFS code into a graph.
	bool toGraph(CGraph &g) const {
		g.clear();
		for(DFSCode::const_iterator it = begin(); it != end(); ++it) {
			g.resize(std::max(it->from, it->to) + 1);
			if(it->fromlabel != (LabelT)-1)
				g[it->from].label = it->fromlabel;
			if(it->tolabel != (LabelT)-1)
				g[it->to].label = it->tolabel;
			g[it->from].push(it->from, it->to, it->elabel);
			if(g.directed == false)
				g[it->to].push(it->to, it->from, it->elabel);
		}
		g.buildEdge();
		return (true);
	}
	// Return number of nodes in the graph.
	unsigned nodeCount(void) {
		unsigned nodecount = 0;
		for(DFSCode::iterator it = begin(); it != end(); ++it)
			nodecount = std::max(nodecount, (unsigned)(std::max(it->from, it->to) + 1));
		return (nodecount);
	}
	DFSCode &operator=(const DFSCode &other) {
		if(this == &other) return *this;
		std::vector<DFS>::operator=(other);
		rmpath = other.rmpath;
		return *this;
	}
	friend bool operator==(const DFSCode &d1, const DFSCode &d2) {
		if(d1.size() != d2.size()) return false;
		for(size_t i = 0; i < d1.size(); i++)
			if(d1[i] != d2[i]) return false;
		return true;
	}
	friend bool operator<(const DFSCode &d1, const DFSCode &d2) {
		if(d1.size() < d2.size()) return true;
		else if(d1.size() > d2.size()) return false;
		for(size_t i = 0; i < d1.size(); i++) {
			if(d1[i] < d2[i]) return true;
			else if(d2[i] < d1[i]) return false;
		}
		return false;         //equal
	}
	friend std::ostream &operator<<(std::ostream &out, const DFSCode &code);
	void push(VeridT from, VeridT to, LabelT fromlabel, LabelT elabel, LabelT tolabel) {
		resize(size() + 1);
		DFS &d = (*this)[size() - 1];
		d.from = from;
		d.to = to;
		d.fromlabel = fromlabel;
		d.elabel = elabel;
		d.tolabel = tolabel;
	}
	void pop() { resize(size() - 1); }
	//std::string to_string(bool print_edge_type = true) const;
	std::string to_string(bool print_edge_type = true) const {
		if (empty()) return "";
		std::stringstream ss;
		size_t i = 0;
		ss << (*this)[i].to_string(print_edge_type);
		i ++;
		for (; i < size(); ++i) {
			ss << ";" << (*this)[i].to_string(print_edge_type);
		}
		return ss.str();
	}

	//std::ostream &write(std::ostream &) const;  // write
	std::ostream &write(std::ostream &os) const {
		if(size() == 0) return os;
		os << "(" << (*this)[0].fromlabel << ") " << (*this)[0].elabel << " (0f" << (*this)[0].tolabel << ");";
		for(unsigned int i = 1; i < size(); ++i) {
			if((*this)[i].from < (*this)[i].to) {
				os << " " << (*this)[i].elabel << " (" << (*this)[i].from << "f" << (*this)[i].tolabel << ");";
			} else {
				os << " " << (*this)[i].elabel << " (b" << (*this)[i].to << ");";
			}
		}
		return os;
	}
/*
	//static DFSCode read_from_str(const std::string &str);
	DFSCode read_from_str(const std::string &str) {
		std::vector<std::string> vec_str;
		split(str, vec_str, ";");
		DFSCode result;
		for(int i = 0; i < vec_str.size(); i++) {
			DFS d = DFS::parse_from_string(vec_str[i].c_str());
			result.push_back(d);
		} // for i
		return result;
	}
	//bool dfs_code_is_min() const;
	//virtual size_t get_serialized_size() const;
	//virtual size_t get_serialized_size(char *buffer, size_t buffer_size) const;
	//virtual size_t serialize(char *buffer, size_t buffer_size) const;
	//virtual size_t deserialize(char *buffer, size_t buffer_size);
	//virtual size_t serialize(std::ostream &) const;
	//virtual size_t deserialize(std::istream &);
	//void remove_negative_ones();
	virtual size_t get_serialized_size() const {
		if(empty()) return sizeof(int);
		return size() * at(0).get_serialized_size() + sizeof(int);
	}
	virtual size_t get_serialized_size(char *buffer, size_t buffer_size) const {
		if(buffer_size < sizeof(int)) throw runtime_error("Buffer too small.");
		return *((int *) buffer);
	}
	virtual size_t serialize(char *buffer, size_t buffer_size) const {
		if(buffer_size < get_serialized_size()) throw runtime_error("Buffer too small.");
		char *buf = buffer;
		size_t stored = 0;
		// store dfs code element count
		*((int*)(buf + stored)) = size();
		stored += sizeof(int);
		// store each dfs element
		for(int i = 0; i < size(); i++) {
			size_t tmp = at(i).serialize(buf + stored, buffer_size - stored);
			stored += tmp;
		} // for i
		return stored;
	} // DFSCode::serialize
	virtual size_t deserialize(char *buffer, size_t buffer_size) {
		if(get_serialized_size(buffer, buffer_size) == 0) return sizeof(int);
		clear();
		int elements = *((int*)buffer);
		size_t read = sizeof(int);
		for(int i = 0; i < elements; i++) {
			DFS d;
			size_t tmp = d.deserialize(buffer + read, buffer_size - read);
			read += tmp;
			push_back(d);
		} // for i
		return read;
	} // DFSCode::deserialize
	virtual size_t serialize(std::ostream &) const {
		throw method_unimplemented("DFSCode::serialize");
	} // DFSCode::serialize
	virtual size_t deserialize(std::istream &) {
		throw method_unimplemented("DFSCode::deserialize");
	} // DFSCode::deserialize
	void remove_negative_ones() {
		if(size() < 1) return;
		int last_vid = at(0).to;
		for(int i = 1; i < size(); i++) {
			if(at(i).from == -1) at(i).from = last_vid;
			if(at(i).is_forward()) last_vid = at(i).to;
		}
	} // DFSCode::remove_negative_ones
//*/
};

std::ostream &operator<<(std::ostream &out, const DFSCode &code) {
  out << code.to_string();
  return out;
}

#endif
