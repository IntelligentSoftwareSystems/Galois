#ifndef SMAT_H
#define SMAT_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <assert.h>

#include <omp.h>
#include <zlib.h>

#define MALLOC(type, size) (type*)malloc(sizeof(type) * (size))

class smat_t;
class entry_iterator_t; // iterator for files with (i,j,v) tuples
class smat_iterator_t;  // iterator for nonzero entries in smat_t

// H = X*W, (X: m*n, W: n*k row-major, H m*k row major)
void smat_x_dmat(const smat_t& X, const double* W, const size_t k, double* H);
// H = a*X*W + H0, (X: m*n, W: n*k row-major, H m*k row major)
void smat_x_dmat(const double a, const smat_t& X, const double* W,
                 const size_t k, const double* H0, double* H);

// Sparse matrix format CCS & RCS
class smat_t {
private:
  bool mem_alloc_by_me;
  bool read_from_binary;
  unsigned char* binary_buf;
  size_t binary_buf_len;
  const static int HeaderSize =
      sizeof(size_t) + sizeof(size_t) + sizeof(size_t) + sizeof(size_t);

public:
  size_t rows, cols;
  long nnz, max_row_nnz, max_col_nnz;
  double *val, *val_t;
  long *col_ptr, *row_ptr;
  long *col_nnz, *row_nnz;
  unsigned *row_idx, *col_idx; // condensed

  // Constructor and Destructor
  smat_t() : mem_alloc_by_me(false), read_from_binary(false) {}
  smat_t(const smat_t& m) {
    *this            = m;
    mem_alloc_by_me  = false;
    read_from_binary = false;
  }
  ~smat_t() { clear_space(); }

  void clear_space();
  smat_t transpose();
  smat_t row_subset(const std::vector<int>& subset);
  smat_t row_subset(const int* subset, int subset_size);

  long nnz_of_row(unsigned i) const { return (row_ptr[i + 1] - row_ptr[i]); }
  long nnz_of_col(unsigned i) const { return (col_ptr[i + 1] - col_ptr[i]); }

  // smat-vector multiplication
  void Xv(const double* v, double* Xv);
  void XTu(const double* u, double* XTu);

  // IO methods
  void load_from_iterator(long _rows, long _cols, long _nnz,
                          entry_iterator_t* entry_it);
  void load(long _rows, long _cols, long _nnz, const char* filename);
  void load_from_binary(const char* filename);
  void save_binary_to_file(const char* filename);

  // temporally methods
  void from_mpi() {
    mem_alloc_by_me = true;
    max_col_nnz     = 0;
    for (size_t c = 1; c <= cols; ++c)
      max_col_nnz = std::max(max_col_nnz, col_ptr[c] - col_ptr[c - 1]);
  }
  void print_mat(int host) {
    for (size_t c = 0; c < cols; ++c)
      if (col_ptr[c + 1] > col_ptr[c]) {
        printf("%ld: %ld at host %d\n", c, col_ptr[c + 1] - col_ptr[c], host);
      }
  }
  double get_global_mean() const;
  void remove_bias(double bias = 0);
};

/*-------------- Iterators -------------------*/

class rate_t {
public:
  unsigned i, j;
  double v;
  rate_t(int ii = 0, int jj = 0, double vv = 0) : i(ii), j(jj), v(vv) {}
};

class entry_iterator_t {
public:
  size_t nnz;
  virtual rate_t next() = 0;
};

// Iterator for files with (i,j,v) tuples
class file_iterator_t : public entry_iterator_t {
public:
  file_iterator_t(size_t nnz_, const char* filename);
  ~file_iterator_t() {
    if (fp)
      fclose(fp);
  }
  rate_t next();

private:
  size_t nnz;
  FILE* fp;
};

// smat_t iterator
class smat_iterator_t : public entry_iterator_t {
public:
  enum { ROWMAJOR, COLMAJOR };
  // major: smat_iterator_t::ROWMAJOR or smat_iterator_t::COLMAJOR
  smat_iterator_t(const smat_t& M, int major = ROWMAJOR);
  ~smat_iterator_t() {}
  rate_t next();

private:
  unsigned* col_idx;
  long* row_ptr;
  double* val_t;
  long rows, cols, cur_idx;
  int cur_row;
};

// smat_t subset iterator
class smat_subset_iterator_t : public entry_iterator_t {
public:
  enum { ROWMAJOR, COLMAJOR };
  // major: smat_iterator_t::ROWMAJOR or smat_iterator_t::COLMAJOR
  smat_subset_iterator_t(const smat_t& M, int const* subset, int size,
                         int major = ROWMAJOR);
  ~smat_subset_iterator_t() {}
  rate_t next();

private:
  unsigned* col_idx;
  long* row_ptr;
  double* val_t;
  long rows, cols, cur_idx;
  int cur_row;
  std::vector<int> subset;
};

#endif
