/*
 * RandomGenerator.h
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */
#include<limits>
#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_
class RandomGenerator{
private:
	long seed;
  static const long multiplier;// = 0x5DEECE66DL;
  static const long addend ;//= 0xBL;
  static const long mask ;//= (1L << 48) - 1;

  /**
   * Creates new RandomGenerator.  You really should set a seed before using it.
   */
public:
  RandomGenerator() {
  }

  RandomGenerator(long inSeed) {
    setSeed(inSeed);
  }

private:
  unsigned long RotateLeft(unsigned long n, unsigned long i){  return (n << i) | (n >> (32 - i));}
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
//    long l = ((long) (nextInt(26)) << 27) + nextInt(27);
//    return l / (double) (1L << 53);
    return d;
  }

public:
  void setSeed(long inSeed) {
    seed = (inSeed ^ multiplier) & mask;
  }


};
const long RandomGenerator::multiplier = 0x5DEECE66DL;
const long RandomGenerator::addend = 0xBL;
const long RandomGenerator::mask = (1L << 48) - 1;
#endif /* RANDOMGENERATOR_H_ */
