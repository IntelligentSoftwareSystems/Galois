/*
 * CUDA BarnesHut v3.1: Simulation of the gravitational forces
 * in a galactic cluster using the Barnes-Hut n-body algorithm
 *
 * Copyright (c) 2013, Texas State University-San Marcos. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted for academic, research, experimental, or personal
 * use provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *    * Neither the name of Texas State University-San Marcos nor the names of
 * its contributors may be used to endorse or promote products derived from this
 *      software without specific prior written permission.
 *
 * For all other uses, please contact the Office for Commercialization and
 * Industry Relations at Texas State University-San Marcos
 * <http://www.txstate.edu/ocir/>.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Martin Burtscher <burtscher@txstate.edu>
 *
 */

#pragma once
#define THREADS1 32
#define THREADS2 192
#define THREADS3 64
#define THREADS4 352
#define THREADS5 192
#define THREADS6 96
#define FACTOR1 6
#define FACTOR2 7
#define FACTOR3 8
#define FACTOR4 4
#define FACTOR5 7
#define FACTOR6 6
static const char* TUNING_PARAMETERS =
    "THREADS1 32\nTHREADS2 192\nTHREADS3 64\nTHREADS4 352\nTHREADS5 "
    "192\nTHREADS6 96\nFACTOR1 6\nFACTOR2 7\nFACTOR3 8\nFACTOR4 4\nFACTOR5 "
    "7\nFACTOR6 6\n";
