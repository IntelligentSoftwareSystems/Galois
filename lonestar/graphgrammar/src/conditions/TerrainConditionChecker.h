#ifndef GALOIS_TERRAINCONDITIONCHECKER_H
#define GALOIS_TERRAINCONDITIONCHECKER_H

#include <values.h>
#include <cmath>
#include "../utils/ConnectivityManager.h"
#include "../utils/utils.h"
#include "../model/Map.h"
#include "../model/ProductionState.h"
#include "../libmgrs/utm.h"
#include "ConditionChecker.h"

//! Uses terrain to determine if a triangle is to be refined.
class TerrainConditionChecker : public ConditionChecker {
public:
  explicit TerrainConditionChecker(double tolerance,
                                   ConnectivityManager& connManager, Map& map)
      : tolerance(tolerance), connManager(connManager), map(map) {}


  //! Only refine if meets inside_condition + is hyperedge node
  bool execute(GNode& node) override {
    NodeData& nodeData = node->getData();
    if (!nodeData.isHyperEdge()) {
      return false;
    }


    // gets coordinates of vertices connected by this hyperedge
    vector<Coordinates> verticesCoords = connManager.getVerticesCoords(node);

    if (!inside_condition(verticesCoords)) {
      return false;
    }

    nodeData.setToRefine(true);
    return true;
  }

private:
  double tolerance;
  ConnectivityManager& connManager;
  Map& map;

  bool inside_condition(
      const vector<Coordinates>&
          verticesCoords) { // TODO: Find better point location algorithm

    // lowest x among 3
    double lowest_x = verticesCoords[0].getX() < verticesCoords[1].getX()
                          ? verticesCoords[0].getX()
                          : verticesCoords[1].getX();
    lowest_x = verticesCoords[2].getX() < lowest_x ? verticesCoords[2].getX()
                                                   : lowest_x;

    // highest x among 3
    double highest_x = verticesCoords[0].getX() > verticesCoords[1].getX()
                           ? verticesCoords[0].getX()
                           : verticesCoords[1].getX();
    highest_x = verticesCoords[2].getX() > highest_x ? verticesCoords[2].getX()
                                                     : highest_x;

    // lowest y among 3
    double lowest_y = verticesCoords[0].getY() < verticesCoords[1].getY()
                          ? verticesCoords[0].getY()
                          : verticesCoords[1].getY();
    lowest_y = verticesCoords[2].getY() < lowest_y ? verticesCoords[2].getY()
                                                   : lowest_y;

    // highest y among 3
    double highest_y = verticesCoords[0].getY() > verticesCoords[1].getY()
                           ? verticesCoords[0].getY()
                           : verticesCoords[1].getY();
    highest_y = verticesCoords[2].getY() > highest_y ? verticesCoords[2].getY()
                                                     : highest_y;

    double step = map.isUtm() ? 90 : map.getCellWidth();
    for (double i = lowest_x; i <= highest_x; i += step) {
      for (double j = lowest_y; j <= highest_y; j += step) {
        Coordinates tmp{i, j, 0.};
        double barycentric_point[3];
        compute_barycentric_coords(barycentric_point, tmp, verticesCoords);
        if (is_inside_triangle(barycentric_point)) {
          double height = 0;
          for (int k = 0; k < 3; ++k) {
            height += barycentric_point[k] * verticesCoords[k].getZ();
          }
          if (fabs(height - map.get_height(i, j)) > tolerance) {
            return true;
          }
        }
      }
    }
    return false;
  }

  void compute_barycentric_coords(double* barycentric_coords,
                                  Coordinates& point,
                                  const vector<Coordinates>& verticesCoords) {
    double triangle_area =
        get_area(verticesCoords[0], verticesCoords[1], verticesCoords[2]);
    barycentric_coords[2] =
        get_area(point, verticesCoords[0], verticesCoords[1]) / triangle_area;
    barycentric_coords[1] =
        get_area(point, verticesCoords[2], verticesCoords[0]) / triangle_area;
    barycentric_coords[0] =
        get_area(point, verticesCoords[1], verticesCoords[2]) / triangle_area;
  }

  bool is_inside_triangle(double barycentric_coords[]) {
    return !greater(barycentric_coords[0] + barycentric_coords[1] +
                        barycentric_coords[2],
                    1.);
  }

  double get_area(const Coordinates& a, const Coordinates& b,
                  const Coordinates& c) {
    return 0.5 * fabs((b.getX() - a.getX()) * (c.getY() - a.getY()) -
                      (b.getY() - a.getY()) * (c.getX() - a.getX()));
  }
};

#endif // GALOIS_TERRAINCONDITIONCHECKER_H
