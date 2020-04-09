#include "Map.h"

#include "../utils/Utils.h"
#include "../libmgrs/utm.h"

#include <cstdio>
#include <cmath>
#include <iostream>
#include <values.h>

double** Map::init_map_data(size_t rows, size_t cols) {
  double** map;
  map = (double**)malloc(rows * sizeof(double*));
  for (size_t i = 0; i < rows; ++i) {
    map[i] = (double*)malloc(cols * sizeof(double));
  }
  return map;
}

void Map::print_map() {
  for (int i = 0; i < this->length; ++i) {
    for (int j = 0; j < this->width; ++j) {
      fprintf(stdout, "%5.0lf ", this->data[i][j]);
    }
    fprintf(stdout, "\n");
  }
}

double Map::get_height(double lon, double lat) {
  return get_height(lon, lat, utm);
}

double Map::get_height(double lon, double lat, bool convert) {

  double x, y;

  // convert to geodetic if required
  if (convert) {
    if (Convert_UTM_To_Geodetic(zone, hemisphere, lon, lat, &y, &x)) {
      fprintf(stderr, "Error during conversion to geodetic.\n");
      exit(18);
    }
    x = Utils::r2d(x);
    y = Utils::r2d(y);
  } else {
    x = lon;
    y = lat;
  }

  // Check if the point is inside the map:
  const auto south_border = north_border - cell_length * length;
  const auto east_border = west_border + cell_width * width;

  if (Utils::is_greater(y, north_border) ||
      Utils::is_lesser(y, south_border)  ||
      Utils::is_greater(x, east_border)  ||
      Utils::is_lesser(x, west_border))  {
	  std::cerr << "Point is outside the map" << std::endl;
	  exit(EXIT_FAILURE);
  }

  // Compute "grid coordinates".
  // modf returns the fractional part of the number,
  // and assigns the integral part to the second argument.
  //
  // The integral part let us know in which "cell" of the map the point is located,
  // and the fractional part let us interpolate the heights.
  double x_grid_int_part, y_grid_int_part;
  const auto y_fract = std::modf((north_border - y) / cell_length, &y_grid_int_part );
  const auto x_fract = std::modf((x - west_border) / cell_width,  &x_grid_int_part);

  // using Lagrange bilinear interpolation
  // Compute the height of the four corners
  double top_left_height     = get_height_wo_interpol(x_grid_int_part, y_grid_int_part, 1);
  double top_right_height    = get_height_wo_interpol(x_grid_int_part, y_grid_int_part, 2);
  double bottom_right_height = get_height_wo_interpol(x_grid_int_part, y_grid_int_part, 3);
  double bottom_left_height  = get_height_wo_interpol(x_grid_int_part, y_grid_int_part, 4);

  // Sum the contributions of each corner
  double height = 0.;
  height += top_left_height * (1 - x_fract) * (1 - y_fract);
  height += top_right_height * x_fract * (1 - y_fract);
  height += bottom_right_height * x_fract * y_fract;
  height += bottom_left_height * (1 - x_fract) * y_fract;

  return height;
}

// corner: 1 - top_left, 2 - top_right, 3 - bottom_right, 4 - bottom_left
double Map::get_height_wo_interpol(const double lon_grid, const double lat_grid, const int corner) {

  auto x = (int)lon_grid; 
  auto y = (int)lat_grid; 

  switch (corner) {
  case 1:
    break;
  case 2:
    ++x;
    break;
  case 3:
    ++x;
    ++y;
    break;
  case 4:
    ++y;
    break;
  default:
    //XXX[AOS]: I think we should raise an error, unless it is used elsewhere.
    return MINDOUBLE;
  }

  return data[y][x];
}

Map::~Map() {
  for (size_t i = 0; i < this->length; ++i) {
    free((double*)this->data[i]);
  }
  free(this->data);
}
