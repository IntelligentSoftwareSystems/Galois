/** Billiards read from simulation log -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Billiards partially ordered version based on applying order independence test to
 * unsorted worklist items 
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#include "Billiards.h"

#include <boost/algorithm/string.hpp>

#include <map>
#include <fstream>
#include <string>
#include <vector>
#include <deque>
#include <sstream>

#include <cstdio>

struct Config {
  // config keys, in the same order as the file
  FP length;
  FP width;
  unsigned num_balls;
  FP ball_mass;
  FP ball_radius;
};

struct BallUpdate {
  unsigned id;
  FP time;
  Vec2 pos;
  Vec2 vel;

  BallUpdate (unsigned id, const FP& time, const Vec2& pos, const Vec2& vel)
    : id (id), time (time), pos (pos), vel (vel) 
  {}

  std::string str () const {
    char s[1024];

    sprintf (s, "BallUpdate: [ball-id=%d,ts=%10.10f,pos=%s,vel=%s]",
        id, time, pos.str ().c_str (), vel.str ().c_str ());

    return s;

  }

};

void splitCSV (std::vector<std::string>& v, const std::string& line, const char* delim=", \t\n") {
  boost::algorithm::split (v, line, boost::algorithm::is_any_of (delim), boost::algorithm::token_compress_on);
}


Config readConfig (const std::string& confName="config.csv") {
  std::ifstream confFile (confName.c_str ());


  assert (confFile.good ());
  std::string header;
  std::getline (confFile, header);


  assert (confFile.good ());
  std::string second;
  std::getline (confFile, second);
  assert (confFile.good ());

  std::vector<std::string> values;
  splitCSV (values, second);

  assert (values.size () == 5);
  Config conf;
  conf.length = std::stod (values[0]);
  conf.width = std::stod (values[1]);
  conf.num_balls = (unsigned) std::stoi (values[2]);
  conf.ball_mass = std::stod (values[3]);
  conf.ball_radius = std::stod (values[4]);

  printf ("conf= length: %g, width=%g, num_balls=%d, ball_mass=%g, ball_radius=%g\n", conf.length, conf.width, conf.num_balls, conf.ball_mass, conf.ball_radius);

  return conf;
}

void readSimLog (std::deque<BallUpdate>& updates, const std::string& logName="simLog.csv") {

  std::ifstream simLog (logName.c_str ());

  std::string line;
  std::vector<std::string> line_tokens;

  // read the header
  std::getline (simLog, line);

  assert (simLog.good ());

  while (simLog.good ()) {

    line_tokens.clear ();
    line.clear ();
    std::getline (simLog, line);

    if (!simLog.good ()) {
      if (simLog.eof ()) { 
        break;

      } else {
        assert (false && "bad state while reading file");
        abort ();
      }
    }

    // std::cout << "line : " << line << std::endl;
    // std::cout << "good: " << simLog.good ()  << std::endl;

    splitCSV (line_tokens, line);

    assert (line_tokens.size () == 6);
    unsigned id = std::stoi (line_tokens[0]);


    FP time = FP (std::stod (line_tokens[1]));
    Vec2 pos (std::stod (line_tokens[2]), std::stod (line_tokens[3]));
    Vec2 vel (std::stod (line_tokens[4]), std::stod (line_tokens[5]));

    updates.push_back (BallUpdate (id, time, pos, vel));

  }
}




int main (int argc, char* argv[]) {

  const Config conf = readConfig ();

  std::deque<BallUpdate> updates;

  readSimLog (updates);

  std::vector<Ball> balls;

  assert (updates.size () >= conf.num_balls);

  // create balls from first num_ball updates;
  for (unsigned i = 0; i < conf.num_balls; ++i) {
    BallUpdate up = updates.front ();
    updates.pop_front ();
    balls.push_back (Ball (up.id, up.pos, up.vel, conf.ball_mass, conf.ball_radius, up.time));

    std::cout << "Ball created: " << balls.back ().str () << std::endl;

  }

  // dummy simulation loop;

  FP total_time = 0.0;
  FP delta = 0.1;

  while (!updates.empty ()) {

    total_time += delta;

    if (total_time >= updates.front ().time) {

      while ((total_time >= updates.front ().time) 
          && (!updates.empty ())) {

        const BallUpdate up = updates.front ();
        updates.pop_front ();

        assert ((balls[up.id].getID () == up.id) && "ball id-index assumption broken"); 
        balls[up.id].update (up.vel, up.time);

        std::cout << "Processing update: " << up.str () << std::endl;

        std::cout << "Ball updated: " << balls[up.id].str () << std::endl;

        // assert (FPutils::almostEqual (up.pos, balls[up.id].pos ()));

      } 
    } 


    std::cout << "--- advancing all balls to: " << total_time << std::endl;
    for (Ball& b: balls) {
      b.update (b.vel (), total_time);
    }

    // check the state of the balls
    for (size_t i = 0; i < balls.size (); ++i) {
      const Ball& b = balls[i];
      const Vec2& p = b.pos ();

      // TODO: include radius for more precise check
      if (p.getX() < DoubleWrapper(0.0) || p.getX () > conf.length) {
        std::cerr << "!!!  ERROR: Ball out of X lim: " << b.str () << std::endl;
      }

      // TODO: include radius for more precise check
      if (p.getY () < DoubleWrapper(0.0) || p.getY () > conf.width) {
        std::cerr << "!!!  ERROR: Ball out of Y lim: " << b.str () << std::endl;
      }

      for (size_t j = i + 1; j < balls.size (); ++i) {
        const Ball& othB = balls[j];

        FP d = p.dist (othB.pos ());
        if (d < (DoubleWrapper(2) * conf.ball_radius)) {
          std::cerr << "!!!  ERROR: Balls overlap: ";
          std::cerr << b.str () << "    ";
          std::cerr << othB.str () << std::endl;
        }
      }

    } // end outer for


  }

  return 0;
}
