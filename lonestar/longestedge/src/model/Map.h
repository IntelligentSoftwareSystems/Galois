#ifndef TERGEN_MAP_H
#define TERGEN_MAP_H

#include <cstdlib>

/**
 * Holds the elevation of a particular point for some specified region (borders
 * and their lengths).
 */
class Map {
private:
  size_t width;

  size_t length;

  double cell_width;

  double cell_length;

  double** data;

  double north_border;

  double west_border;

  bool utm; // TODO: Consider moving it to some better place

  long zone;

  char hemisphere;

  double get_height_wo_interpol(const double lon, const double lat, const int corner);

public:
  Map(double** data, size_t width, size_t length, double cellWidth,
      double cellLength)
      : width(width), length(length), cell_width(cellWidth),
        cell_length(cellLength), data(data), utm(true) {}

  static double** init_map_data(size_t rows, size_t cols);

  void print_map();

  double get_height(double lon, double lat);

  double get_height(double lon, double lat, bool convert);

  size_t getWidth() const { return width; }

  void setWidth(size_t width) { Map::width = width; }

  size_t getLength() const { return length; }

  void setLength(size_t length) { Map::length = length; }

  double getCellWidth() const { return cell_width; }

  void setCellWidth(double cellWidth) { cell_width = cellWidth; }

  double getCellLength() const { return cell_length; }

  void setCellLength(double cellLength) { cell_length = cellLength; }

  double** getData() const { return data; }

  void setData(double** data) { Map::data = data; }

  double getNorthBorder() const { return north_border; }

  void setNorthBorder(double northBorder) { north_border = northBorder; }

  double getWestBorder() const { return west_border; }

  void setWestBorder(double westBorder) { west_border = westBorder; }

  bool isUtm() const { return utm; }

  void setUtm(bool utm) { Map::utm = utm; }

  long getZone() const { return zone; }

  void setZone(long zone) { Map::zone = zone; }

  char getHemisphere() const { return hemisphere; }

  void setHemisphere(char hemisphere) { Map::hemisphere = hemisphere; }

  ~Map();
};

#endif // TERGEN_MAP_H
