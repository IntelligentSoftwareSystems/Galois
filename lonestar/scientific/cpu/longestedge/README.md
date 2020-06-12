Longest Edge
============

DESCRIPTION 
-----------

This program runs a variant of Rivaras mesh refinement algorithm on portions of the earth's surface.
It requires the data available at https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/ and can generate `.node`, `.ele.`, and `.poly` files that follow the same format used in https://www.cs.cmu.edu/~quake/triangle.html.
The command line inputs are the bounds of a box in UTM coordinates.

BUILD
-----

1. Run cmake at BUILD directory (refer to top-level README for cmake instructions).

2. Run `cd <BUILD>/lonestar/scientific/cpu/longestedge && make -j`


RUN
---

The following is an example command line call:

 - `./longestedge -l 25 -s 14 -N 52.4 -S 49. -E 23.1 -W 18.1 -data <dataDirectory> -o <outputFile> -square -altOutput`

