#ifndef EQUATION_OPTIMIZER_H
#define EQUATION_OPTIMIZER_H

#include "../functional/BitVectorPool.h"
#include "../functional/FunctionHandler.h"
#include "../factoring/EquationData.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <tuple>
#include <functional>
#include <cstdlib>
#include <algorithm>

namespace Factoring {

typedef unsigned long int word;

typedef struct bucketElement {
	word* function;
	unsigned int support;
	unsigned short int ccDist;
	EquationData equation;

	bucketElement() : function( nullptr ), support( 0 ), ccDist( 0 ) { }

} BucketElement;

typedef struct literalDisapearanceData {
	bool negCof_negLit;
	bool negCof_posLit;
	bool posCof_negLit;
	bool posCof_posLit;

	literalDisapearanceData() {
		negCof_negLit = false;
		negCof_posLit = false;
		posCof_negLit = false;
		posCof_posLit = false;
	}

} LiteralDisapearanceData;

typedef std::vector< std::string > StringVector;
typedef std::vector< short > ShortVector;
typedef std::vector< bool > BoolVector;
typedef std::unordered_map< std::string, std::pair< word*, unsigned int > > StringFunctionMap;
typedef std::vector< EquationData > EquationDataVector;
using FunctionSet = std::unordered_set< word*, Functional::FunctionHasher, Functional::FunctionComparator >;
typedef std::vector< BucketElement > Bucket;
typedef std::vector < Bucket > BucketVector;
typedef Functional::BitVectorPool FunctionPool;

class EquationOptimizer {

	word * targetFunction;
	StringVector & varSet;
	StringFunctionMap & literals;
	std::unordered_set< std::string > usedLiterals;
	FunctionPool & functionPool;

	int nVars;
	int nBits;
	int nWords;
	int nBucketsSTL;

	Functional::FunctionHasher functionHasher;
	Functional::FunctionComparator functionComparator;
	FunctionSet cubeCofactors;
	FunctionSet cubeCofactorsComb;
	FunctionSet alreadyVisited;
	BucketElement lookAheadSolution;
	Functional::FunctionDataMap cubeCofactorData;
	LiteralDisapearanceData ** literalDisapearanceMatrix;

	EquationDataVector solutions;
	BucketVector smallerBuckets;
	BucketVector largerBuckets;
	BucketVector notComparableBuckets;

	bool runningLookAhead;
	bool xorEnabled;
	bool muxEnabled;
	bool verbose;
	bool stop;
	short currentNumberOfSolutions;
	short desiredNumberOfSolutions;
	int iBucket;
	long int nCC;
	long int nCCcomb;
	double killGraphRuntime;
	double cubeCofactorRuntime;
	double cubeCofactorCombRuntime;
	double bucketCombRuntime;

	void createLiterals();
	void computeLiteralDisapearanceByCofactors();
	void computeLiteralDisapearance( unsigned int supportCof0, unsigned int supportCof1, int iVar );
	void computeInitialBuckets();
	void combineAllCubeCofactors();
	void combineAllCubeCofactorData();
	void computeCurrentBucket();
	void lookAhead();

	void combineBucketsSMxSM( int i, int j );
	void combineBucketsLGxLG( int i, int j );
	void combineBucketsNCxNC( int i, int j );
	void combineBucketsSMxNC( int i, int j );
	void combineBucketsLGxNC( int i, int j );
	void combineThreeBuckets( const BucketVector & first, const BucketVector & second, const BucketVector & third, int i, int j, int k );

	void AND( const BucketElement & e1, const BucketElement & e2 );
	void OR( const BucketElement & e1, const BucketElement & e2 );
	void XOR( const BucketElement & e1, const BucketElement & e2 );
	void MUX( const BucketElement & e1, const BucketElement & e2, const BucketElement & e3 );

	bool abandon( BucketElement & element );
	void insertInBuckets( BucketElement & element );
	void registerSolution( EquationData & solution );

public:
      
	EquationOptimizer( word * targetFunction, std::vector< String > & varSet, StringFunctionMap & literals, FunctionPool & functionPool, int maxNumSolutions );
	~EquationOptimizer();
	void optimizeEquation();
	EquationDataVector & getSolutions();
	void printSolutions();
	void printBucket( Bucket & bucket, std::string label );
	void printLiteralDisapearanceMatrix();

	int getIBucket();
	long int getNCC();
	long int getNCCcomb();
	double getKillGraphRuntime();
	double getCubeCofactorRuntime();
	double getCubeCofactorCombRuntime();
	double getBucketCombRuntime();
};

} // namespace Factoring

#endif
