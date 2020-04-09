#include <cstdio>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <galois/gIO.h>
#include "SrtmReader.h"
#include "../utils/Utils.h"

Map* SrtmReader::read(
    const double west_border, const double north_border,
    const double east_border, const double south_border,
    const char* map_dir) { // data[row][column] - it's array of rows
  Utils::swap_if_required((double*)&south_border, (double*)&north_border);
  Utils::swap_if_required((double*)&west_border, (double*)&east_border);

  // Rounding to avoid problems with numerical errors
  // Adding some margin to make sure that the mesh is always inside the map
  int west_border_int  = border_to_int(west_border) - MARGIN;
  int north_border_int = border_to_int(north_border) + MARGIN;
  int east_border_int  = border_to_int(east_border) + MARGIN;
  int south_border_int = border_to_int(south_border) - MARGIN;

  // Update the map vertices
  const auto map_N_border = (double)north_border_int / VALUES_IN_DEGREE;
  const auto map_S_border = (double)south_border_int / VALUES_IN_DEGREE;
  const auto map_E_border = (double)east_border_int / VALUES_IN_DEGREE;
  const auto map_W_border = (double)west_border_int / VALUES_IN_DEGREE;

  size_t cols       = (size_t)(east_border_int - west_border_int);
  size_t rows       = (size_t)(north_border_int - south_border_int);
  // mallocs an entire grid of doubles based on borders; to be read in later
  // from disk
  double** map_data = Map::init_map_data(rows, cols);
  Map* map          = new Map(map_data, cols, rows, 1. / VALUES_IN_DEGREE,
                     1. / VALUES_IN_DEGREE);


  map->setNorthBorder(map_N_border);
  map->setWestBorder(map_W_border);

  // read in points within specified borders
  read_from_multiple_files(map_W_border, map_N_border, map_E_border, map_S_border,
                           map_dir, map_data);

  skip_outliers(map_data, map->getLength(), map->getWidth());

  return map;
}

void SrtmReader::read_from_multiple_files(const double west_border,
                                          const double north_border,
                                          const double east_border,
                                          const double south_border,
                                          const char* map_dir,
                                          double** map_data) {
  int first_free_row = 0;
  double north_ptr   = north_border;
  double south_ptr = Utils::is_lesser(Utils::floor2(north_border), south_border)
                         ? south_border
                         : Utils::floor2(north_border);
  if (Utils::equals(north_ptr, south_ptr)) {
    south_ptr = Utils::is_lesser(north_border - 1, south_border)
                    ? south_border
                    : north_border - 1;
  }

  // loop over y-axis
  while (Utils::is_greater(north_ptr, south_border)) {
    int north_ptr_int = border_to_int(north_ptr);
    size_t rows_here =
        (size_t)std::abs(north_ptr_int - border_to_int(south_ptr));

    int first_free_col = 0;
    double west_ptr    = west_border;
    double east_ptr = Utils::is_greater(Utils::ceil2(west_border), east_border)
                          ? east_border
                          : Utils::ceil2(west_border);

    // loop over x-axis
    while (Utils::is_lesser(west_ptr, east_border)) {
      int west_ptr_int = border_to_int(west_ptr);
      size_t cols_here =
          (size_t)std::abs(border_to_int(east_ptr) - west_ptr_int);
      // determine file to read and actual do read
      read_from_file(north_ptr_int, west_ptr_int, rows_here, cols_here,
                     first_free_row, first_free_col, map_data, map_dir);

      first_free_col += cols_here;
      west_ptr = Utils::floor2(west_ptr + 1);
      east_ptr = Utils::is_greater(east_ptr + 1, east_border) ? east_border
                                                              : east_ptr + 1;
    }

    first_free_row += rows_here;
    north_ptr = Utils::equals(Utils::floor2(north_ptr), north_ptr)
                    ? north_ptr - 1
                    : Utils::floor2(north_ptr);
    south_ptr = Utils::is_lesser(south_ptr - 1, south_border) ? south_border
                                                              : south_ptr - 1;
  }
}

void SrtmReader::read_from_file(int north_border_int, int west_border_int,
                                size_t rows, size_t cols, int first_row,
                                int first_col, double** map_data,
                                const char* map_dir) {
  char file_to_open[256];
  get_filename(file_to_open, map_dir, west_border_int, north_border_int);

  FILE* map_file;
  if ((map_file = fopen(file_to_open, "rb")) == NULL) {
    fprintf(stderr, "%s\n", strerror(errno));
    exit(1);
  }
  int cells_in_degree = VALUES_IN_DEGREE + 1;
  if (fseek(map_file,
            (((VALUES_IN_DEGREE - (north_border_int % VALUES_IN_DEGREE)) %
              VALUES_IN_DEGREE) *
                 cells_in_degree +
             (west_border_int % VALUES_IN_DEGREE)) *
                PIXEL_SIZE,
            SEEK_SET) == -1) {
    fprintf(stderr, "%s\n", strerror(errno));
    exit(1);
  }
  uint16_t* buffer = (uint16_t*)malloc(PIXEL_SIZE * cols);
  for (size_t i = 0; i < rows; ++i) {
    fread(buffer, PIXEL_SIZE, cols, map_file);
    if (fseek(map_file, (cells_in_degree - cols) * PIXEL_SIZE, SEEK_CUR) ==
        -1) {
      fprintf(stderr, "%s\n", strerror(errno));
      exit(1);
    }
    for (size_t j = 0; j < cols; ++j) {
      Utils::change_bytes_order(&(buffer[j]));
      // here is where data is being written after being read
      map_data[first_row + i][first_col + j] = buffer[j];
    }
  }
  free(buffer);
  if (fclose(map_file) != 0) {
    fprintf(stderr, "%s\n", strerror(errno));
    exit(1);
  }
}

void SrtmReader::skip_outliers(double* const* map_data, size_t length,
                               size_t width) {
  bool outlierFound = false;
  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < width; ++j) {
      // smooth out outlier point here
      if (map_data[i][j] > 3000 || map_data[i][j] < 10) {
        outlierFound = true;
        if (i > 0) {
          map_data[i][j] = map_data[i - 1][j];
        } else {
          map_data[i][j] = map_data[i + 1][j];
        }
      }
    }
  }

  if (outlierFound) {
    galois::gInfo("Outliers in input data detected.");
  }
}

void SrtmReader::get_filename(char* filename, const char* map_dir,
                              int west_border_int, int north_border_int) {
  int first_long_to_read;
  int first_lat_to_read;

  if (west_border_int < 0) {
    if (west_border_int % VALUES_IN_DEGREE != 0) {
      first_long_to_read = west_border_int / VALUES_IN_DEGREE + 1;
    } else {
      first_long_to_read = west_border_int / VALUES_IN_DEGREE;
    }
  } else {
    first_long_to_read = west_border_int / VALUES_IN_DEGREE;
  }

  if (north_border_int < 0) {
    if (north_border_int % VALUES_IN_DEGREE != 0) {
      first_lat_to_read = north_border_int / VALUES_IN_DEGREE + 1;
    } else {
      first_lat_to_read = north_border_int / VALUES_IN_DEGREE;
    }
  } else {
    if (north_border_int % VALUES_IN_DEGREE != 0) {
      first_lat_to_read = north_border_int / VALUES_IN_DEGREE;
    } else {
      first_lat_to_read = north_border_int / VALUES_IN_DEGREE - 1;
    }
  }

  sprintf(filename, "%s/%s%d%s%.3d.hgt", map_dir,
          first_lat_to_read < 0 ? "S" : "N", first_lat_to_read,
          first_long_to_read < 0 ? "W" : "E", first_long_to_read);
}

int SrtmReader::border_to_int(const double border) {
  return (int)round(border * SrtmReader::VALUES_IN_DEGREE);
}
