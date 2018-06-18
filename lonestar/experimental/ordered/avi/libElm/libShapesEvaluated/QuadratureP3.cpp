#include "QuadratureP3.h"

const double Tet_4Point::BulkCoordinates[] = {
    0.58541020e0, 0.13819660e0, 0.13819660e0, 0.13819660e0,
    0.58541020e0, 0.13819660e0, 0.13819660e0, 0.13819660e0,
    0.58541020e0, 0.13819660e0, 0.13819660e0, 0.13819660e0};

const double Tet_4Point::BulkWeights[] = {1. / 24., 1. / 24., 1. / 24.,
                                          1. / 24.};

const double Tet_4Point::FaceMapCoordinates[] = {2. / 3., 1. / 6., 1. / 6.,
                                                 2. / 3., 1. / 6., 1. / 6.};

// Face 1 : 2-1-0.
const double Tet_4Point::FaceOneShapeCoordinates[] = {
    1. / 6., 1. / 6., 0., 1. / 6., 2. / 3., 0., 2. / 3., 1. / 6., 0.};

const double Tet_4Point::FaceOneWeights[] = {1. / 6., 1. / 6., 1. / 6.};

// Face 2 : 2-0-3.
const double Tet_4Point::FaceTwoShapeCoordinates[] = {
    1. / 6., 0., 1. / 6., 2. / 3., 0., 1. / 6., 1. / 6., 0., 2. / 3.};

const double Tet_4Point::FaceTwoWeights[] = {1. / 6., 1. / 6., 1. / 6.};

// Face 3: 2-3-1.
const double Tet_4Point::FaceThreeShapeCoordinates[] = {
    0., 1. / 6., 1. / 6., 0., 1. / 6., 2. / 3., 0., 2. / 3., 1. / 6.};

const double Tet_4Point::FaceThreeWeights[] = {1. / 6., 1. / 6., 1. / 6.};

// Face 4: 0-1-3.
const double Tet_4Point::FaceFourShapeCoordinates[] = {
    2. / 3., 1. / 6., 1. / 6., 1. / 6., 2. / 3.,
    1. / 6., 1. / 6., 1. / 6., 2. / 3.};

const double Tet_4Point::FaceFourWeights[] = {1. / 6., 1. / 6., 1. / 6.};

const Quadrature* const Tet_4Point::Bulk =
    new Quadrature(Tet_4Point::BulkCoordinates, Tet_4Point::BulkWeights, 3, 4);

const Quadrature* const Tet_4Point::FaceOne = new Quadrature(
    Tet_4Point::FaceMapCoordinates, Tet_4Point::FaceOneShapeCoordinates,
    Tet_4Point::FaceOneWeights, 2, 3, 3);

const Quadrature* const Tet_4Point::FaceTwo = new Quadrature(
    Tet_4Point::FaceMapCoordinates, Tet_4Point::FaceTwoShapeCoordinates,
    Tet_4Point::FaceTwoWeights, 2, 3, 3);

const Quadrature* const Tet_4Point::FaceThree = new Quadrature(
    Tet_4Point::FaceMapCoordinates, Tet_4Point::FaceThreeShapeCoordinates,
    Tet_4Point::FaceThreeWeights, 2, 3, 3);

const Quadrature* const Tet_4Point::FaceFour = new Quadrature(
    Tet_4Point::FaceMapCoordinates, Tet_4Point::FaceFourShapeCoordinates,
    Tet_4Point::FaceFourWeights, 2, 3, 3);

const double Tet_11Point::BulkCoordinates[] = {1. / 4.,
                                               1. / 4.,
                                               1. / 4.,
                                               11. / 14.,
                                               1. / 14.,
                                               1. / 14.,
                                               1. / 14.,
                                               11. / 14.,
                                               1. / 14.,
                                               1. / 14.,
                                               1. / 14.,
                                               11. / 14.,
                                               1. / 14.,
                                               1. / 14.,
                                               1. / 14.,
                                               0.3994035761667992,
                                               0.3994035761667992,
                                               0.1005964238332008,
                                               0.3994035761667992,
                                               0.1005964238332008,
                                               0.3994035761667992,
                                               0.3994035761667992,
                                               0.1005964238332008,
                                               0.1005964238332008,
                                               0.1005964238332008,
                                               0.3994035761667992,
                                               0.3994035761667992,
                                               0.1005964238332008,
                                               0.3994035761667992,
                                               0.1005964238332008,
                                               0.1005964238332008,
                                               0.1005964238332008,
                                               0.3994035761667992};

const double Tet_11Point::BulkWeights[] = {
    -74. / 5625.,  343. / 45000., 343. / 45000., 343. / 45000.,
    343. / 45000., 56. / 2250.,   56. / 2250.,   56. / 2250.,
    56. / 2250.,   56. / 2250.,   56. / 2250.};

const Quadrature* const Tet_11Point::Bulk = new Quadrature(
    Tet_11Point::BulkCoordinates, Tet_11Point::BulkWeights, 3, 11);

const double Tet_15Point::BulkCoordinates[] = {
    1. / 4.,           1. / 4.,           1. / 4.,           0.,
    1. / 3.,           1. / 3.,           1. / 3.,           0.,
    1. / 3.,           1. / 3.,           1. / 3.,           0.,
    1. / 3.,           1. / 3.,           1. / 3.,           72. / 99.,
    1. / 11.,          1. / 11.,          1. / 11.,          72. / 99.,
    1. / 11.,          1. / 11.,          1. / 11.,          72. / 99.,
    1. / 11.,          1. / 11.,          1. / 11.,          0.066550153573664,
    0.066550153573664, 0.433449846426336, 0.066550153573664, 0.433449846426336,
    0.066550153573664, 0.066550153573664, 0.433449846426336, 0.433449846426336,
    0.433449846426336, 0.066550153573664, 0.066550153573664, 0.433449846426336,
    0.433449846426336, 0.066550153573664, 0.433449846426336, 0.066550153573664,
    0.433449846426336};

const double Tet_15Point::BulkWeights[] = {
    0.030283678097089, 0.006026785714286, 0.006026785714286, 0.006026785714286,
    0.006026785714286, 0.011645249086029, 0.011645249086029, 0.011645249086029,
    0.011645249086029, 0.010949141561386, 0.010949141561386, 0.010949141561386,
    0.010949141561386, 0.010949141561386, 0.010949141561386};

const Quadrature* const Tet_15Point::Bulk = new Quadrature(
    Tet_15Point::BulkCoordinates, Tet_15Point::BulkWeights, 3, 15);
// Sriramajayam
