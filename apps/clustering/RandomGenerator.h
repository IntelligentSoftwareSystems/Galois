/*
 * RandomGenerator.h
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */
#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_
#include <limits>
#include <stdint.h>
class RandomGenerator{
private:
  uint64_t seed;
  static const uint64_t multiplier;// = 0x5DEECE66DL;
  static const uint64_t addend ;//= 0xBL;
  static const uint64_t mask ;//= (1L << 48) - 1;

  /**
   * Creates new RandomGenerator.  You really should set a seed before using it.
   */
public:
  RandomGenerator() {
  }

  RandomGenerator(uint64_t inSeed) {
    setSeed(inSeed);
  }

private:
  uint64_t RotateLeft(uint64_t n, uint64_t i){  return (n << i) | (n >> (32 - i));}
  int nextInt(int bits) {
	  return rand();
//    seed = (seed * multiplier + addend) & mask;
//    int ret = seed;
//    return RotateLeft(ret, 48-bits);
//    return (int) (seed >>> (48 - bits));
//    return ret;
  }

public:
  double nextDouble() {
	  double d = (double(rand()))/std::numeric_limits<int>::max();
//    uint64_t l = ((uint64_t) (nextInt(26)) << 27) + nextInt(27);
//    return l / (double) (1L << 53);
    return d;
  }

public:
  void setSeed(uint64_t inSeed) {
    seed = (inSeed ^ multiplier) & mask;
  }


};
const uint64_t RandomGenerator::multiplier = 0x5DEECE66DULL;
const uint64_t RandomGenerator::addend = 0xBULL;
const uint64_t RandomGenerator::mask = (1ULL << 48) - 1;
#endif /* RANDOMGENERATOR_H_ */
