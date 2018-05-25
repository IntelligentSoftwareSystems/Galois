/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "Element.h"
#include "Point.h"

std::ostream& operator<<(std::ostream& out, const Element& e) {
  return e.print(out);
}

bool Element::inTriangle(const Tuple& p) const {
  if (boundary())
    return false;

  const Tuple& p1 = points[0]->t();
  const Tuple& p2 = points[1]->t();
  const Tuple& p3 = points[2]->t();

  if ((p1 == p) || (p2 == p) || (p3 == p)) {
    return false;
  }

  int count = 0;
  double px = p.x();
  double py = p.y();
  double p1x = p1.x();
  double p1y = p1.y();
  double p2x = p2.x();
  double p2y = p2.y();
  double p3x = p3.x();
  double p3y = p3.y();

  if (p2x < p1x) {
    if ((p2x < px) && (p1x >= px)) {
      if (((py - p2y) * (p1x - p2x)) < ((px - p2x) * (p1y - p2y))) {
        count = 1;
      }
    }
  } else {
    if ((p1x < px) && (p2x >= px)) {
      if (((py - p1y) * (p2x - p1x)) < ((px - p1x) * (p2y - p1y))) {
        count = 1;
      }
    }
  }

  if (p3x < p2x) {
    if ((p3x < px) && (p2x >= px)) {
      if (((py - p3y) * (p2x - p3x)) < ((px - p3x) * (p2y - p3y))) {
        if (count == 1) {
          return false;
        }
        count++;
      }
    }
  } else {
    if ((p2x < px) && (p3x >= px)) {
      if (((py - p2y) * (p3x - p2x)) < ((px - p2x) * (p3y - p2y))) {
        if (count == 1) {
          return false;
        }
        count++;
      }
    }
  }

  if (p1x < p3x) {
    if ((p1x < px) && (p3x >= px)) {
      if (((py - p1y) * (p3x - p1x)) < ((px - p1x) * (p3y - p1y))) {
        if (count == 1) {
          return false;
        }
        count++;
      }
    }
  } else {
    if ((p3x < px) && (p1x >= px)) {
      if (((py - p3y) * (p1x - p3x)) < ((px - p3x) * (p1y - p3y))) {
        if (count == 1) {
          return false;
        }
        count++;
      }
    }
  }

  return count == 1;
}

bool Element::clockwise() const {
  assert(!boundary());

  double t1_x = points[0]->t().x();
  double t1_y = points[0]->t().y();

  double t2_x = points[1]->t().x();
  double t2_y = points[1]->t().y();

  double t3_x = points[2]->t().x();
  double t3_y = points[2]->t().y();

  double counter_clockwise = (t2_x - t1_x) * (t3_y - t1_y) - (t3_x - t1_x) * (t2_y - t1_y);

  return counter_clockwise < 0;
}

bool Element::inCircle(const Tuple& p) const {
  if (boundary())
    return false;

  // This version computes the determinant of a matrix including the
  // coordinates of each points + distance of these points to the origin
  // in order to check if a point is inside a triangle or not
  double t1_x = points[0]->t().x();
  double t1_y = points[0]->t().y();

  double t2_x = points[1]->t().x();
  double t2_y = points[1]->t().y();

  double t3_x = points[2]->t().x();
  double t3_y = points[2]->t().y();

  double p_x = p.x();
  double p_y = p.y();

  // Check if the points (t1,t2,t3) are sorted clockwise or
  // counter-clockwise:
  // -> counter_clockwise > 0 => counter clockwise
  // -> counter_clockwise = 0 => degenerated triangle
  // -> counter_clockwise < 0 => clockwise
  double counter_clockwise = (t2_x - t1_x) * (t3_y - t1_y) - (t3_x - t1_x) * (t2_y - t1_y);

  // If the triangle is degenerate, then the triangle should be updated
  if (counter_clockwise == 0.0) {
    return true;
  }

  // Compute the following determinant:
  // | t1_x-p_x  t1_y-p_y  (t1_x-p_x)^2+(t1_y-p_y)^2 |
  // | t2_x-p_x  t2_y-p_y  (t2_x-p_x)^2+(t2_y-p_y)^2 |
  // | t3_x-p_x  t3_y-p_y  (t3_x-p_x)^2+(t3_y-p_y)^2 |
  //
  // If the determinant is >0 then the point (p_x,p_y) is inside the
  // circumcircle of the triangle (t1,t2,t3).

  // Value of columns 1 and 2 of the matrix
  double t1_p_x, t1_p_y, t2_p_x, t2_p_y, t3_p_x, t3_p_y;
  // Determinant of minors extracted from columns 1 and 2
  // (det_t3_t1_m corresponds to the opposite)
  double det_t1_t2, det_t2_t3, det_t3_t1_m;
  // Values of the column 3 of the matrix
  double t1_col3, t2_col3, t3_col3;

  t1_p_x = t1_x - p_x;
  t1_p_y = t1_y - p_y;
  t2_p_x = t2_x - p_x;
  t2_p_y = t2_y - p_y;
  t3_p_x = t3_x - p_x;
  t3_p_y = t3_y - p_y;

  det_t1_t2 = t1_p_x * t2_p_y - t2_p_x * t1_p_y;
  det_t2_t3 = t2_p_x * t3_p_y - t3_p_x * t2_p_y;
  det_t3_t1_m = t3_p_x * t1_p_y - t1_p_x * t3_p_y;
  t1_col3 = t1_p_x * t1_p_x + t1_p_y * t1_p_y;
  t2_col3 = t2_p_x * t2_p_x + t2_p_y * t2_p_y;
  t3_col3 = t3_p_x * t3_p_x + t3_p_y * t3_p_y;

  double det = t1_col3 * det_t2_t3 + t2_col3 * det_t3_t1_m + t3_col3 * det_t1_t2;

  // If the points are enumerated in clockwise, then negate the result
  if (counter_clockwise < 0) {
    return det < 0;
  }
  return det > 0;
}

std::ostream& Element::print(std::ostream& out) const {
  out << '[';
  for (int i = 0; i < dim(); ++i) {
    out << points[i]->id() << " ";
    points[i]->print(out);
    out << (i < (dim() - 1) ? ", " : "");
  }
  out << ']';
  return out;
}

