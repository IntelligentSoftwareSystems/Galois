#ifndef TERGEN_SRTMREADER_H
#define TERGEN_SRTMREADER_H

#include <cstdlib>
#include "../model/Map.h"

class SrtmReader {
private:
  static const int RESOLUTION = 3;

  static const unsigned short PIXEL_SIZE = 2;

  /**
   * Given the border of the coordinates to read, read files corresponding to
   * the points within the border
   *
   * @param map_dir directory storing the files to read
   * @param map_data 2D malloc'd array representing the map
   */
  void read_from_multiple_files(const double west_border,
                                const double north_border,
                                const double east_border,
                                const double south_border, const char* map_dir,
                                double** map_data);

  /**
   * Given north and west starting point as well as rows/columns to read,
   * find the file to read and read it into the map
   */
  void read_from_file(int north_border_int, int west_border_int, size_t rows,
                      size_t cols, int first_row, int first_col,
                      double** map_data, const char* map_dir);

  /**
   * Smooth out read outliers by making them take a nearby point.
   */
  void skip_outliers(double* const* map_data, size_t length, size_t width);

  //! Given north and west points, determine file name to read
  void get_filename(char* filename, const char* map_dir, int west_border_int,
                    int north_border_int);

  //! Convert a border point into an int
  int border_to_int(const double border);

public:
  static const int VALUES_IN_DEGREE = 60 * 60 / RESOLUTION;
  static const int MARGIN = 3;

  Map* read(const double west_border, const double north_border,
            const double east_border, const double south_border,
            const char* map_dir);
};

#endif // TERGEN_SRTMREADER_H
