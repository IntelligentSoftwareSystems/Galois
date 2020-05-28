#include "../../src/utils/Utils.h"
#include "../../src/model/Map.h"

TEST_CASE("convertToUtm Test") {
  double longitude     = 20.;
  double latitude      = 50.;
  long zone            = 34;
  char hemisphere      = 'N';
  double northing      = 5539109.82;
  double easting       = 428333.55;
  double** placeholder = (double**)malloc(sizeof(double*));
  placeholder[0]       = (double*)malloc(sizeof(double));
  placeholder[0][0]    = 8;
  Map map{placeholder, 1, 1, 1., 1.};

  const std::pair<double, double>& pair =
      Utils::convertToUtm(latitude, longitude, map);

  REQUIRE(fabs(pair.first - easting) < 1e-1);
  REQUIRE(fabs(pair.second - northing) < 1e-1);
  REQUIRE(map.getZone() == zone);
  REQUIRE(map.getHemisphere() == hemisphere);
}