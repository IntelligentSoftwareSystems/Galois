/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#include "font8x8.h"

const int WIDTH  = 1024;
const int HEIGHT = 1024;

struct rgbData {
  uint8_t red;
  uint8_t green;
  uint8_t blue;
};

static void setPixel(struct rgbData data[][WIDTH], int x, int y,
                     struct rgbData color) {
  if (x < 0 || x >= WIDTH)
    return;
  if (y < 0 || y >= HEIGHT)
    return;
  data[y][x] = color;
}

static void setPixelUnsafe(struct rgbData data[][WIDTH], int x, int y,
                           struct rgbData color) {
  data[y][x] = color;
}

static void drawchar(struct rgbData data[][WIDTH], int x0, int y0, char d,
                     struct rgbData color) {
  int set;
  // int mask;
  unsigned char* bitmap = font8x8_basic[(int)d];
  for (int x = 0; x < 8; x++) {
    for (int y = 0; y < 8; y++) {
      set = bitmap[x] & 1 << y;
      if (set)
        setPixel(data, x0 + y, y0 + x, color);
    }
  }
}

static inline void drawstring(struct rgbData data[][WIDTH], int x0, int y0,
                              const char* d, struct rgbData color) {
  for (int i = 0; i < strlen(d); ++i)
    drawchar(data, x0 + i * 8, y0, d[i], color);
}

static int sgn(int x) { return ((x < 0) ? -1 : ((x > 0) ? 1 : 0)); }

static void drawlineUnsafe(rgbData data[][WIDTH], int x1, int y1, int x2,
                           int y2, struct rgbData color) {
  int dx    = x2 - x1;
  int dy    = y2 - y1;
  int dxabs = abs(dx);
  int dyabs = abs(dy);
  int sdx   = sgn(dx);
  int sdy   = sgn(dy);
  int x     = dyabs >> 1;
  int y     = dxabs >> 1;
  int px    = x1;
  int py    = y1;
  if (dxabs >= dyabs) {
    for (int i = 0; i < dxabs; ++i) {
      y += dyabs;
      if (y >= dxabs) {
        y -= dxabs;
        py += sdy;
      }
      px += sdx;
      setPixelUnsafe(data, px, py, color);
    }
  } else {
    for (int i = 0; i < dyabs; ++i) {
      x += dxabs;
      if (x >= dyabs) {
        x -= dyabs;
        px += sdx;
      }
      py += sdy;
      setPixelUnsafe(data, px, py, color);
    }
  }
}

typedef int OutCode;

const int INSIDE = 0; // 0000
const int LEFT   = 1; // 0001
const int RIGHT  = 2; // 0010
const int BOTTOM = 4; // 0100
const int TOP    = 8; // 1000

// Compute the bit code for a point (x, y) using the clip rectangle
// bounded diagonally by (xmin, ymin), and (xmax, ymax)
OutCode ComputeOutCode(int x, int y) {
  OutCode code = INSIDE; // initialised as being inside of clip window
  if (x < 0)             // to the left of clip window
    code |= LEFT;
  else if (x >= WIDTH) // to the right of clip window
    code |= RIGHT;
  if (y < 0) // below the clip window
    code |= BOTTOM;
  else if (y >= HEIGHT) // above the clip window
    code |= TOP;
  return code;
}

// Cohen–Sutherland clipping algorithm clips a line from
// P0 = (x0, y0) to P1 = (x1, y1) against a rectangle with
// diagonal from (xmin, ymin) to (xmax, ymax).
void drawline(rgbData data[][WIDTH], int x0, int y0, int x1, int y1,
              struct rgbData color) {
  // compute outcodes for P0, P1, and whatever point lies outside the clip
  // rectangle
  OutCode outcode0 = ComputeOutCode(x0, y0);
  OutCode outcode1 = ComputeOutCode(x1, y1);
  bool accept      = false;

  while (true) {
    if (!(outcode0 |
          outcode1)) { // Bitwise OR is 0. Trivially accept and get out of loop
      accept = true;
      break;
    } else if (outcode0 & outcode1) { // Bitwise AND is not 0. Trivially reject
                                      // and get out of loop
      break;
    } else {
      // failed both tests, so calculate the line segment to clip
      // from an outside point to an intersection with clip edge
      double x, y;

      // At least one endpoint is outside the clip rectangle; pick it.
      OutCode outcodeOut = outcode0 ? outcode0 : outcode1;

      // Now find the intersection point;
      // use formulas y = y0 + slope * (x - x0), x = x0 + (1 / slope) * (y - y0)
      if (outcodeOut & TOP) { // point is above the clip rectangle
        x = x0 + (x1 - x0) * (HEIGHT - 1 - y0) / (double)(y1 - y0);
        y = HEIGHT - 1;
      } else if (outcodeOut & BOTTOM) { // point is below the clip rectangle
        x = x0 + (x1 - x0) * (0 - y0) / (double)(y1 - y0);
        y = 0;
      } else if (outcodeOut &
                 RIGHT) { // point is to the right of clip rectangle
        y = y0 + (y1 - y0) * (WIDTH - 1 - x0) / (double)(x1 - x0);
        x = WIDTH - 1;
      } else if (outcodeOut & LEFT) { // point is to the left of clip rectangle
        y = y0 + (y1 - y0) * (0 - x0) / (double)(x1 - x0);
        x = 0;
      }

      // Now we move outside point to intersection point to clip
      // and get ready for next pass.
      if (outcodeOut == outcode0) {
        x0       = (int)x;
        y0       = (int)y;
        outcode0 = ComputeOutCode(x0, y0);
      } else {
        x1       = (int)x;
        y1       = (int)y;
        outcode1 = ComputeOutCode(x1, y1);
      }
    }
  }
  if (accept)
    drawlineUnsafe(data, x0, y0, x1, y1, color);
}

static void drawcircle(struct rgbData data[][WIDTH], int x0, int y0, int radius,
                       rgbData color) {
  int f     = 1 - radius;
  int ddF_x = 1;
  int ddF_y = -2 * radius;
  int x     = 0;
  int y     = radius;

  setPixel(data, x0, y0 + radius, color);
  setPixel(data, x0, y0 - radius, color);
  setPixel(data, x0 + radius, y0, color);
  setPixel(data, x0 - radius, y0, color);

  while (x < y) {
    // ddF_x == 2 * x + 1;
    // ddF_y == -2 * y;
    // f == x*x + y*y - radius*radius + 2*x - y + 1;
    if (f >= 0) {
      y--;
      ddF_y += 2;
      f += ddF_y;
    }
    x++;
    ddF_x += 2;
    f += ddF_x;
    setPixel(data, x0 + x, y0 + y, color);
    setPixel(data, x0 - x, y0 + y, color);
    setPixel(data, x0 + x, y0 - y, color);
    setPixel(data, x0 - x, y0 - y, color);
    setPixel(data, x0 + y, y0 + x, color);
    setPixel(data, x0 - y, y0 + x, color);
    setPixel(data, x0 + y, y0 - x, color);
    setPixel(data, x0 - y, y0 - x, color);
  }
}
