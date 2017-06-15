#include "../factoring/EquationOptimizer.h"

namespace Factoring {

#define CC_DIST 1

EquationOptimizer::EquationOptimizer( word * targetFunction, std::vector< String > & varSet, StringFunctionMap & literals, FunctionPool & fucntionPool, int maxNumSolutions ) :
		targetFunction( targetFunction ),
		varSet( varSet ),
		literals( literals ),
		functionPool( fucntionPool ),
		nVars( varSet.size() ),
		nBits( pow( 2, nVars ) ),
		nWords( Functional::wordNum( nVars ) ),
		nBucketsSTL( nVars * nWords * 100 ),
		functionHasher( nWords ),
		functionComparator( nWords ),
		cubeCofactors( nBucketsSTL, functionHasher, functionComparator ),
		cubeCofactorsComb( nBucketsSTL, functionHasher, functionComparator ),
		alreadyVisited( nBucketsSTL, functionHasher, functionComparator ),
		cubeCofactorData( nBucketsSTL, functionHasher, functionComparator ),
		smallerBuckets( 64 ),
		largerBuckets( 64 ),
		notComparableBuckets( 64 ),
		desiredNumberOfSolutions( maxNumSolutions )
	{

	// Allocate and initialize the matrix to store literals disapearance information
	LiteralDisapearanceData * tmp = (LiteralDisapearanceData*) malloc( sizeof( LiteralDisapearanceData )*nVars*nVars );
	this->literalDisapearanceMatrix = (LiteralDisapearanceData**) malloc( sizeof( LiteralDisapearanceData* )*nVars );
	for ( int i = 0, j = 0; i < nVars; i++ ) {
		this->literalDisapearanceMatrix[i] = &tmp[j];
		j += nVars;
	}
	for ( int iVar = 0; iVar < nVars; iVar++ ) {
		for ( int jVar = 0; jVar < nVars; jVar++ ) {
			this->literalDisapearanceMatrix[ iVar ][ jVar ] = LiteralDisapearanceData();
		}
	}

	this->lookAheadSolution.function = nullptr;
	this->lookAheadSolution.ccDist = 0;
	this->lookAheadSolution.support = 0;
	this->lookAheadSolution.equation = EquationData( "", 0, 0, 0 );

	this->runningLookAhead = false;
	this->xorEnabled = false;
	this->muxEnabled = false;
	this->verbose = false;
	this->stop = false;

	this->iBucket = 1;
	this->nCC = 0;
	this->nCCcomb = 0;
	this->currentNumberOfSolutions = 0;
	this->desiredNumberOfSolutions = maxNumSolutions;
	this->killGraphRuntime = 0;
	this->cubeCofactorRuntime = 0;
	this->cubeCofactorCombRuntime = 0;
	this->bucketCombRuntime = 0;

	if ( this->verbose ) {
		std::cout << "Target Function: " << Functional::toHex( this->targetFunction, nWords ) << std::endl;
	}
}

EquationOptimizer::~EquationOptimizer() {
	free( this->literalDisapearanceMatrix[0] );
	free( this->literalDisapearanceMatrix );
}

void EquationOptimizer::optimizeEquation() {

	double startTime, endTime;

	startTime = clock();
//	computeLiteralDisapearanceByCofactors();
//	computeInitialBuckets();
	endTime = clock();
	this->killGraphRuntime = (endTime - startTime) / (double) CLOCKS_PER_SEC;

	startTime = clock();
	Functional::computeAllCubeCofactors( this->functionPool, this->cubeCofactors, this->targetFunction, this->nVars );
	//Functional::computeAllCubeCofactorsWithSupport( this->functionPool, this->cubeCofactorData, this->targetFunction, this->nVars );
	this->nCC = cubeCofactors.size();
	endTime = clock();
	this->cubeCofactorRuntime = (endTime - startTime) / (double) CLOCKS_PER_SEC;

	createLiterals();

	if ( this->stop ) {
		return;
	}

	startTime = clock();
	combineAllCubeCofactors();
	//combineAllCubeCofactorData();
	this->nCCcomb = cubeCofactors.size();
	endTime = clock();	
	this->cubeCofactorCombRuntime = (endTime - startTime) / (double) CLOCKS_PER_SEC;

	// To avoid constants in the buckets
	this->alreadyVisited.insert( this->literals["0"].first );
	this->alreadyVisited.insert( this->literals["1"].first );

	int firstBucket = 2;
//	if ( (!this->smallerBuckets[2].empty()) || (!this->largerBuckets[2].empty()) || (!this->notComparableBuckets[2].empty()) ) {
//		firstBucket = 3;
//	}

	if ( this->verbose ) {
		std::cout << "Buckets " << 1 << ":" << std::endl;
		printBucket( this->smallerBuckets[1], "SMALLER" );
		printBucket( this->largerBuckets[1], "LARGER" );
		printBucket( this->notComparableBuckets[1], "NOTCOMPARABLE" );

		if ( firstBucket == 3 ) {
			std::cout << "Buckets " << 2 << ":" << std::endl;
			printBucket( this->smallerBuckets[2], "SMALLER" );
			printBucket( this->largerBuckets[2], "LARGER" );
			printBucket( this->notComparableBuckets[2], "NOTCOMPARABLE" );
		}
	}

	startTime = clock();
	for ( this->iBucket = firstBucket; this->iBucket < 64; this->iBucket++ ) { // FIXME, why 64?

		computeCurrentBucket();

		if ( this->verbose ) {
			std::cout << "Buckets " << this->iBucket << ":" << std::endl;
			printBucket( this->smallerBuckets[ this->iBucket ], "SMALLER" );
			printBucket( this->largerBuckets[ this->iBucket ], "LARGER" );
			printBucket( this->notComparableBuckets[ this->iBucket ], "NOTCOMPARABLE" );
		}

		lookAhead();

		if ( this->lookAheadSolution.equation.getLiterals() > 0 ) {
			registerSolution( lookAheadSolution.equation );
			if ( this->solutions.size() == this->desiredNumberOfSolutions ) {
				this->stop = true;
				break;
			}
		}

		if ( this->stop ) {
			break;
		}
	}

	endTime = clock();	
	this->bucketCombRuntime = (endTime - startTime) / (double) CLOCKS_PER_SEC;

	//std::cout << "DONE" << std::endl;
}

void EquationOptimizer::createLiterals() {

	std::string lit;
	int binateVars = 0;
	bool posVar;
	bool negVar;

	for ( int iVar = 0; iVar < nVars; iVar++ ) {

		posVar = Functional::posVar( this->targetFunction, this->nVars, iVar );
		negVar = Functional::negVar( this->targetFunction, this->nVars, iVar );

		if ( ( posVar ) || ( !posVar && !negVar ) ) {
			lit = this->varSet[ iVar ];
			BucketElement element;
			element.function = this->literals[ lit ].first;
			element.support = this->literals[ lit ].second;
			if ( this->cubeCofactors.count( element.function ) != 0 ) {
				element.ccDist = 0;
			}
			else {
				element.ccDist = 1;
			}
			element.equation = EquationData( lit, 1, 1, 1 );
			insertInBuckets( element );
		}

		if ( ( negVar ) || ( !posVar && !negVar ) ) {
			std::string lit = this->varSet[ iVar ];
			lit = "!" + lit;
			BucketElement element;
			element.function = this->literals[ lit ].first;
			element.support = this->literals[ lit ].second;
			if ( this->cubeCofactors.count( element.function ) != 0 ) {
				element.ccDist = 0;
			}
			else {
				element.ccDist = 1;
			}
			element.equation = EquationData( lit, 1, 1, 1 );
			insertInBuckets( element );
		}

		if ( !posVar && !negVar ) {
			binateVars++;
		}
	}

	if ( this->xorEnabled && (binateVars < 2) ) {
		this->xorEnabled = false; // Para gerar XOR eh preciso pelo menos duas variaveis binate
	}

	if ( this->muxEnabled && (binateVars < 1) ) {
		this->muxEnabled = false; // Para gerar MUX eh preciso pelo menos uma variavel binate
	}
}

void EquationOptimizer::computeLiteralDisapearanceByCofactors() {

	for ( int iVar = 0; iVar < nVars; iVar++ ) {
		// Compute cofactor with iVar = 0
		word * negCof = functionPool.getMemory();
		Functional::cofactor0( negCof, this->targetFunction, this->nWords, iVar );
		unsigned int supportCof0 = Functional::getPolarizedSupport( negCof, this->nVars );

		// Compute cofactor with iVar = 1
		word * posCof = functionPool.getMemory();
		Functional::cofactor1( posCof, this->targetFunction, this->nWords, iVar );
		unsigned int supportCof1 = Functional::getPolarizedSupport( posCof, this->nVars );

		computeLiteralDisapearance( supportCof0, supportCof1, iVar );
	}
	//printLiteralDisapearanceMatrix();
}


void EquationOptimizer::computeLiteralDisapearance( unsigned int supportCof0, unsigned int supportCof1, int iVar ) {

	for ( int jVar = 0; jVar < this->nVars; jVar++ ) {

		if ( iVar != jVar ) {

			bool negVar = Functional::negVar( this->targetFunction, this->nVars, jVar );
			bool posVar = Functional::posVar( this->targetFunction, this->nVars, jVar );

			if ( ( negVar ) || ( !posVar && !negVar ) ) {
				std::string lit = "!" + this->varSet[ jVar ];
				unsigned int negLitSup = this->literals[ lit ].second;
				if ( (supportCof0 & negLitSup) == 0 ) {
					// Cofator negativo de iVar matou o literal negativo de jVar
					this->literalDisapearanceMatrix[ iVar ][ jVar ].negCof_negLit = true;
				}
				if ( (supportCof1 & negLitSup) == 0 ) {
					// Cofator positivo de iVar matou o literal negativo de jVar
					this->literalDisapearanceMatrix[ iVar ][ jVar ].posCof_negLit = true;
				}
				this->usedLiterals.insert( lit );
			}

			if ( ( posVar ) || ( !posVar && !negVar ) ) {
				std::string lit = this->varSet[ jVar ];
				unsigned int posLitSup = this->literals[ lit ].second;
				if ( (supportCof0 & posLitSup) == 0 ) {
					// Cofator negativo de iVar matou o literal positivo de jVar
					this->literalDisapearanceMatrix[ iVar ][ jVar ].negCof_posLit = true;
				}
				if ( (supportCof1 & posLitSup) == 0 ) {
					// Cofator positivo de iVar matou o literal positivo de jVar
					this->literalDisapearanceMatrix[ iVar ][ jVar ].posCof_posLit = true;
				}
				this->usedLiterals.insert( lit );
			}
		}
	}
}

void EquationOptimizer::computeInitialBuckets() {

	std::string piLit, niLit, pjLit, njLit;

	for ( int iVar = 0; iVar < this->nVars; iVar++ ) {

		for ( int jVar = iVar+1; jVar < this->nVars; jVar++ ) {

			if ( iVar != jVar ) {

				LiteralDisapearanceData iVarTOjVar = this->literalDisapearanceMatrix[ iVar ][ jVar ];
				LiteralDisapearanceData jVarTOiVar = this->literalDisapearanceMatrix[ jVar ][ iVar ];
				piLit = this->varSet[ iVar ];
				niLit = "!" + piLit;
				pjLit = this->varSet[ jVar ];
				njLit = "!" + pjLit;

				if ( iVarTOjVar.posCof_posLit && jVarTOiVar.posCof_posLit ) {
					// iLit + jLit
					BucketElement element;
					std::string eq = "(" + piLit + "+" + pjLit + ")";
					element.equation = EquationData( eq, 1, 0, 2 );
					element.support = this->literals[ piLit ].second | this->literals[ pjLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::OR( element.function, this->literals[ piLit ].first, this->literals[ pjLit ].first, this->nWords );

					insertInBuckets( element );
				}
				if ( iVarTOjVar.negCof_posLit && jVarTOiVar.posCof_negLit ) {
					// !iLit + jLit
					BucketElement element;
					std::string eq = "(" + niLit + "+" + pjLit + ")";
					element.equation = EquationData( eq, 0, 1, 2 );
					element.support = this->literals[ niLit ].second | this->literals[ pjLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::OR( element.function, this->literals[ niLit ].first, this->literals[ pjLit ].first, this->nWords );
					insertInBuckets( element );
				}
				if ( iVarTOjVar.posCof_negLit && jVarTOiVar.negCof_posLit ) {
					// iLit + !jLit
					BucketElement element;
					std::string eq = "(" + piLit + "+" + njLit + ")";
					element.equation = EquationData( eq, 0, 1, 2 );
					element.support = this->literals[ piLit ].second | this->literals[ njLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::OR( element.function, this->literals[ piLit ].first, this->literals[ njLit ].first, this->nWords );
					insertInBuckets( element );
				}
				if ( iVarTOjVar.negCof_negLit && jVarTOiVar.negCof_negLit ) {
					// !iLit + !jLit
					BucketElement element;
					std::string eq = "(" + niLit + "+" + njLit + ")";
					element.equation = EquationData( eq, 0, 1, 2 );
					element.support = this->literals[ niLit ].second | this->literals[ njLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::OR( element.function, this->literals[ niLit ].first, this->literals[ njLit ].first, this->nWords );
					insertInBuckets( element );
				}
				if ( iVarTOjVar.negCof_posLit && jVarTOiVar.negCof_posLit ) {
					// iLit * jLit
					BucketElement element;
					std::string eq = "(" + piLit + "*" + pjLit + ")";
					element.equation = EquationData( eq, 1, 0, 2 );
					element.support = this->literals[ piLit ].second | this->literals[ pjLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::AND( element.function, this->literals[ piLit ].first, this->literals[ pjLit ].first, this->nWords );
					insertInBuckets( element );
				}
				if ( iVarTOjVar.posCof_posLit && jVarTOiVar.negCof_negLit ) {
					// !iLit * jLit
					BucketElement element;
					std::string eq = "(" + niLit + "*" + pjLit + ")";
					element.equation = EquationData( eq, 1, 0, 2 );
					element.support = this->literals[ niLit ].second | this->literals[ pjLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::AND( element.function, this->literals[ niLit ].first, this->literals[ pjLit ].first, this->nWords );
					insertInBuckets( element );
				}
				if ( iVarTOjVar.negCof_negLit && jVarTOiVar.posCof_posLit ) {
					// iLit * !jLit
					BucketElement element;
					std::string eq = "(" + piLit + "*" + njLit + ")";
					element.equation = EquationData( eq, 1, 0, 2 );
					element.support = this->literals[ piLit ].second | this->literals[ njLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::AND( element.function, this->literals[ piLit ].first, this->literals[ njLit ].first, this->nWords );
					insertInBuckets( element );
				}
				if ( iVarTOjVar.posCof_negLit && jVarTOiVar.posCof_negLit ) {
					// !iLit * !jLit
					BucketElement element;
					std::string eq = "(" + niLit + "*" + njLit + ")";
					element.equation = EquationData( eq, 1, 0, 2 );
					element.support = this->literals[ niLit ].second | this->literals[ njLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::AND( element.function, this->literals[ niLit ].first, this->literals[ njLit ].first, this->nWords );
					insertInBuckets( element );
				}
				if ( iVarTOjVar.negCof_negLit && iVarTOjVar.posCof_posLit && jVarTOiVar.negCof_negLit && jVarTOiVar.posCof_posLit ) {
					// iLit ^ jLit
					BucketElement element;
					std::string eq = "(" + piLit + "^" + pjLit + ")";
					element.equation = EquationData( eq, 1, 1, 2 );
					element.support = this->literals[ piLit ].second | this->literals[ pjLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::XOR( element.function, this->literals[ piLit ].first, this->literals[ pjLit ].first, this->nWords );
					insertInBuckets( element );
				}
				if ( iVarTOjVar.negCof_posLit && iVarTOjVar.posCof_negLit && jVarTOiVar.negCof_posLit && jVarTOiVar.posCof_negLit ) {
					// !(iLit ^ jLit)
					BucketElement element;
					std::string eq = "!(" + piLit + "^" + pjLit + ")";
					element.equation = EquationData( eq, 1, 1, 2 );
					element.support = this->literals[ piLit ].second | this->literals[ pjLit ].second;
					element.function = this->functionPool.getMemory();
					Functional::XOR( element.function, this->literals[ piLit ].first, this->literals[ pjLit ].first, this->nWords );
					Functional::NOT( element.function, element.function, this->nWords );
					insertInBuckets( element );
				}
			}
		}
	}
}

void EquationOptimizer::combineAllCubeCofactors() {

	this->cubeCofactors.erase( this->targetFunction );

	for ( auto element : this->cubeCofactors ) {
		this->cubeCofactorsComb.insert( element );
	}

	word *f1, *f2, *f3, *fr;
	//FunctionSet newlyCreatedFunctions( this->nBucketsSTL, this->functionHasher, this->functionComparator );
	FunctionSet::iterator it_i, it_j, it_k;

	for ( it_i = cubeCofactors.begin(); it_i != cubeCofactors.end(); it_i++ ) {

		f1 = (*it_i);

		for ( it_j = std::next(it_i, 1); it_j != cubeCofactors.end(); it_j++ ) {

			f2 = (*it_j);

			Functional::Order orderf1 = Functional::order( f1, this->targetFunction, this->nWords );
			Functional::Order orderf2 = Functional::order( f2, this->targetFunction, this->nWords );
	
			if ( ( (orderf1 == Functional::Order::SMALLER) && (orderf2 == Functional::Order::SMALLER) ) ||
				 ( (orderf1 == Functional::Order::NOTCOMPARABLE) && (orderf2 == Functional::Order::SMALLER) ) ||
				 ( (orderf1 == Functional::Order::SMALLER) && (orderf2 == Functional::Order::NOTCOMPARABLE) ) ) {

				// OR
				fr = this->functionPool.getMemory();
				Functional::OR( fr, f1, f2, this->nWords );
				cubeCofactorsComb.insert( fr );
				//newlyCreatedFunctions.insert( fr );
				//std::cout << Functional::toHex( fr, this->nWords ) << " = " << Functional::toHex( f1, this->nWords ) << "+" << Functional::toHex( f2, this->nWords ) << std::endl;
			}

			if ( ( (orderf1 == Functional::Order::LARGER) && (orderf2 == Functional::Order::LARGER) ) ||
				 ( (orderf1 == Functional::Order::NOTCOMPARABLE) && (orderf2 == Functional::Order::LARGER) ) ||
				 ( (orderf1 == Functional::Order::LARGER) && (orderf2 == Functional::Order::NOTCOMPARABLE) ) ) {

				// AND
				fr = this->functionPool.getMemory();
				Functional::AND( fr, f1, f2, this->nWords );
				cubeCofactorsComb.insert( fr );
				//newlyCreatedFunctions.insert( fr );
				//std::cout << Functional::toHex( fr, this->nWords ) << " = " << Functional::toHex( f1, this->nWords ) << "*" << Functional::toHex( f2, this->nWords ) << std::endl;
			}

			if ( (orderf1 == Functional::Order::NOTCOMPARABLE) && (orderf2 == Functional::Order::NOTCOMPARABLE) ) {

				// OR
				fr = this->functionPool.getMemory();
				Functional::OR( fr, f1, f2, this->nWords );
				cubeCofactorsComb.insert( fr );
				//newlyCreatedFunctions.insert( fr );
				//std::cout << Functional::toHex( fr, this->nWords ) << " = " << Functional::toHex( f1, this->nWords ) << "+" << Functional::toHex( f2, this->nWords ) << std::endl;

				// AND
				fr = this->functionPool.getMemory();
				Functional::AND( fr, f1, f2, this->nWords );
				cubeCofactorsComb.insert( fr );
				//newlyCreatedFunctions.insert( fr );
				//std::cout << Functional::toHex( fr, this->nWords ) << " = " << Functional::toHex( f1, this->nWords ) << "*" << Functional::toHex( f2, this->nWords ) << std::endl;

				// XOR
				if ( this->xorEnabled ) {
					fr = this->functionPool.getMemory();
					Functional::XOR( fr, f1, f2, this->nWords );
					cubeCofactorsComb.insert( fr );
					//newlyCreatedFunctions.insert( fr );
				}
			}

			if ( this->muxEnabled ) {

				for ( it_k = std::next(it_j, 1); it_k != cubeCofactors.end(); it_k++ ) {

					f3 = (*it_k);

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f1, f2, f3, this->nWords );
					cubeCofactorsComb.insert( fr );
					//newlyCreatedFunctions.insert( fr );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f2, f1, f3, this->nWords );
					cubeCofactorsComb.insert( fr );
					//newlyCreatedFunctions.insert( fr );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f1, f3, f2, this->nWords );
					cubeCofactorsComb.insert( fr );
					//newlyCreatedFunctions.insert( fr );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f3, f1, f2, this->nWords );
					cubeCofactorsComb.insert( fr );
					//newlyCreatedFunctions.insert( fr );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f2, f3, f1, this->nWords );
					cubeCofactorsComb.insert( fr );
					//newlyCreatedFunctions.insert( fr );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f3, f2, f1, this->nWords );
					cubeCofactorsComb.insert( fr );
					//newlyCreatedFunctions.insert( fr );
				}
			}
		}
	}

	// Merge the cubeCofactors and the newlyCreated sets
//	for ( it_i = newlyCreatedFunctions.begin(); it_i != newlyCreatedFunctions.end(); it_i++ ) {
//		this->cubeCofactors.insert( (*it_i) );
//	}

	cubeCofactors.insert( this->targetFunction );
	cubeCofactorsComb.insert( this->targetFunction );
}

void EquationOptimizer::combineAllCubeCofactorData() {

	cubeCofactorData.erase( this->targetFunction );

	word *f1, *f2, *f3, *fr;
	Functional::FunctionDataMap newlyCreatedFunctions( this->nBucketsSTL, this->functionHasher, this->functionComparator );
	Functional::FunctionDataMap::iterator it_i, it_j, it_k;

	for ( it_i = cubeCofactorData.begin(); it_i != cubeCofactorData.end(); it_i++ ) {

		f1 = it_i->first;

		for ( it_j = std::next(it_i, 1); it_j != cubeCofactorData.end(); it_j++ ) {

			f2 = it_j->first;

			// OR
			fr = this->functionPool.getMemory();
			Functional::OR( fr, f1, f2, this->nWords );
			Functional::registerFunction( newlyCreatedFunctions, fr, this->nVars, 1 );

			// AND
			fr = this->functionPool.getMemory();
			Functional::AND( fr, f1, f2, this->nWords );
			Functional::registerFunction( newlyCreatedFunctions, fr, this->nVars, 1 );

			// XOR
			if ( this->xorEnabled ) {
				fr = this->functionPool.getMemory();
				Functional::XOR( fr, f1, f2, this->nWords );
				Functional::registerFunction( newlyCreatedFunctions, fr, this->nVars, 1 );
			}

			if ( this->muxEnabled ) {

				for ( it_k = std::next(it_j, 1); it_k != cubeCofactorData.end(); it_k++ ) {

					f3 = it_k->first;

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f1, f2, f3, this->nWords );
					Functional::registerFunction( newlyCreatedFunctions, fr, this->nVars, 1 );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f2, f1, f3, this->nWords );
					Functional::registerFunction( newlyCreatedFunctions, fr, this->nVars, 1 );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f1, f3, f2, this->nWords );
					Functional::registerFunction( newlyCreatedFunctions, fr, this->nVars, 1 );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f3, f1, f2, this->nWords );
					Functional::registerFunction( newlyCreatedFunctions, fr, this->nVars, 1 );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f2, f3, f1, this->nWords );
					Functional::registerFunction( newlyCreatedFunctions, fr, this->nVars, 1 );

					fr = this->functionPool.getMemory();
					Functional::MUX( fr, f3, f2, f1, this->nWords );
					Functional::registerFunction( newlyCreatedFunctions, fr, this->nVars, 1 );
				}
			}
		}
	}

	// Merge the cubeCofactors and the newlyCreated sets
	for ( it_i = newlyCreatedFunctions.begin(); it_i != newlyCreatedFunctions.end(); it_i++ ) {
		this->cubeCofactorData.insert( (*it_i) );
	}

	Functional::registerFunction( this->cubeCofactorData, this->targetFunction, this->nVars, 1 );
}

void EquationOptimizer::computeCurrentBucket() {

	int i = this->iBucket;

	// AND, OR and XOR (if enabled)
	for ( int fewerLit = 1; fewerLit <= (i/2); fewerLit++ ) {
		if ( this->verbose ) std::cout << std::endl << "x: " << fewerLit << " y: " << (i-fewerLit) << std::endl;
		combineBucketsSMxSM( fewerLit, (i-fewerLit) );
		if ( this->stop ) return;
		combineBucketsLGxLG( fewerLit, (i-fewerLit) );
		if ( this->stop ) return;
		combineBucketsNCxNC( fewerLit, (i-fewerLit) );
		if ( this->stop ) return;
		combineBucketsSMxNC( fewerLit, (i-fewerLit) );
		if ( this->stop ) return;
		combineBucketsLGxNC( fewerLit, (i-fewerLit) );
		if ( this->stop ) return;
	}

	// MUX (if enabled)
	if ( this->muxEnabled && i > 2 ) {
		int y, complement;
		for ( int x=i-2; x >= 1; x-- ) {
			complement = (i-x);
			for ( int z = 1; z <= (complement/2); z++ ) {
				y = (complement-z);
				if ( x >= y && y >= z ) { // Used to avoid repeated combinations of buckets
					if ( this->verbose ) std::cout << std::endl << "x: " << x << " y: " << y << " z: " << z << std::endl;
					combineThreeBuckets( this->notComparableBuckets, this->notComparableBuckets, this->notComparableBuckets, x, y, z );
					if ( this->stop ) return;
					combineThreeBuckets( this->smallerBuckets, this->notComparableBuckets, this->notComparableBuckets, x, y, z );
					if ( this->stop ) return;
					combineThreeBuckets( this->largerBuckets, this->notComparableBuckets, this->notComparableBuckets, x, y, z );
					if ( this->stop ) return;
					// TODO what about other combinations?
				}
			}
		}
	}
}

void EquationOptimizer::lookAhead() {
	this->runningLookAhead = true;
	for ( int i = 1; i <= this->iBucket; i++ ) {
		for ( int j = 1; j <= this->iBucket; j++ ) {
			combineBucketsSMxSM( i, j );
			combineBucketsLGxLG( i, j );
		}
	}
	this->runningLookAhead = false;
}

void EquationOptimizer::combineBucketsSMxSM( int i, int j ) {

	const Bucket & bucket_i = this->smallerBuckets[i];
	const Bucket & bucket_j = this->smallerBuckets[j];
	Bucket::const_iterator it_i, it_j;

	if ( this->verbose ) std::cout << bucket_i.size() << "x" << bucket_j.size() << std::endl;

	if ( bucket_i.empty() || bucket_j.empty() ) {
		return;
	}

	if ( i == j ) {
		for ( it_i = bucket_i.begin(); it_i != bucket_i.end(); it_i++ ) {
			for ( it_j = std::next(it_i, 1); it_j != bucket_i.end(); it_j++ ) {
				OR( (*it_i), (*it_j) );
				if ( this->stop ) return;
			}
		}
	}
	else {
		for ( it_i = bucket_i.begin(); it_i != bucket_i.end(); it_i++ ) {
			for ( it_j = bucket_j.begin(); it_j != bucket_j.end(); it_j++ ) {
				OR( (*it_i), (*it_j) );
				if ( this->stop ) return;
			}
		}
	}
}

void EquationOptimizer::combineBucketsLGxLG( int i, int j ) {

	const Bucket & bucket_i = this->largerBuckets[i];
	const Bucket & bucket_j = this->largerBuckets[j];
	Bucket::const_iterator it_i, it_j;

	if ( this->verbose ) std::cout << bucket_i.size() << "x" << bucket_j.size() << std::endl;

	if ( bucket_i.empty() || bucket_j.empty() ) {
		return;
	}

	if ( i == j ) {
		for ( it_i = bucket_i.begin(); it_i != bucket_i.end(); it_i++ ) {
			for ( it_j = std::next(it_i, 1); it_j != bucket_i.end(); it_j++ ) {
				AND( (*it_i), (*it_j) );
				if ( this->stop ) return;
			}
		}
	}
	else {
		for ( it_i = bucket_i.begin(); it_i != bucket_i.end(); it_i++ ) {
			for ( it_j = bucket_j.begin(); it_j != bucket_j.end(); it_j++ ) {
				AND( (*it_i), (*it_j) );
				if ( this->stop ) return;
			}
		}
	}
}

void EquationOptimizer::combineBucketsNCxNC( int i, int j ) {

	const Bucket & bucket_i = this->notComparableBuckets[i];
	const Bucket & bucket_j = this->notComparableBuckets[j];
	Bucket::const_iterator it_i, it_j;

	if ( this->verbose ) std::cout << bucket_i.size() << "x" << bucket_j.size() << std::endl;

	if ( bucket_i.empty() || bucket_j.empty() ) {
		return;
	}

	if ( i == j ) {
		for ( it_i = bucket_i.begin(); it_i != bucket_i.end(); it_i++ ) {
			for ( it_j = std::next(it_i, 1); it_j != bucket_i.end(); it_j++ ) {

				AND( (*it_i), (*it_j) );
				if ( this->stop ) return;

				OR( (*it_i), (*it_j) );
				if ( this->stop ) return;

				if( this->xorEnabled ) {
					XOR( (*it_i), (*it_j) );
					if ( this->stop ) return;
				}
			}
		}
	}
	else {
		for ( it_i = bucket_i.begin(); it_i != bucket_i.end(); it_i++ ) {
			for ( it_j = bucket_j.begin(); it_j != bucket_j.end(); it_j++ ) {

				AND( (*it_i), (*it_j) );
				if ( this->stop ) return;

				OR( (*it_i), (*it_j) );
				if ( this->stop ) return;

				if( this->xorEnabled ) {
					XOR( (*it_i), (*it_j) );
					if ( this->stop ) return;
				}
			}
		}
	}
}

void EquationOptimizer::combineBucketsSMxNC( int i, int j ) {

	if ( (this->smallerBuckets[i].empty() == false) && (this->notComparableBuckets[j].empty() == false) ) {
		for ( auto element_i : this->smallerBuckets[i] ) {
			for ( auto element_j : this->notComparableBuckets[j] ) {
				AND( element_i, element_j );
				if ( this->stop ) return;

				OR( element_i, element_j );
				if ( this->stop ) return;
			}
		}
	}

	if ( (this->smallerBuckets[j].empty() == false) && (this->notComparableBuckets[i].empty() == false) ) {
		for ( auto element_i : this->notComparableBuckets[i] ) {
			for ( auto element_j : this->smallerBuckets[j] ) {
				AND( element_i, element_j );
				if ( this->stop ) return;

				OR( element_i, element_j );
				if ( this->stop ) return;
			}
		}
	}
}

void EquationOptimizer::combineBucketsLGxNC( int i, int j ) {

	if ( (this->largerBuckets[i].empty() == false) && (this->notComparableBuckets[j].empty() == false) ) {
		for ( auto element_i : this->largerBuckets[i] ) {
			for ( auto element_j : this->notComparableBuckets[j] ) {
				AND( element_i, element_j );
				if ( this->stop ) return;

				OR( element_i, element_j );
				if ( this->stop ) return;
			}
		}
	}

	if ( (this->largerBuckets[j].empty() == false) && (this->notComparableBuckets[i].empty() == false) ) {
		for ( auto element_i : this->notComparableBuckets[i] ) {
			for ( auto element_j : this->largerBuckets[j] ) {
				AND( element_i, element_j );
				if ( this->stop ) return;

				OR( element_i, element_j );
				if ( this->stop ) return;
			}
		}
	}
}

void EquationOptimizer::combineThreeBuckets( const BucketVector & first, const BucketVector & second, const BucketVector & third, int i, int j, int k ) {

	const Bucket & bucket_i = first[i];
	const Bucket & bucket_j = second[j];
	const Bucket & bucket_k = third[k];
	Bucket::const_iterator it_i, it_j, it_k;

	if ( this->verbose ) std::cout << bucket_i.size() << "x" << bucket_j.size() << "x" << bucket_k.size() << std::endl;

	if ( bucket_i.empty() || bucket_j.empty() || bucket_k.empty() ) {
		return;
	}

	for ( it_i = bucket_i.begin(); it_i != bucket_i.end(); it_i++ ) {
		for ( it_j = bucket_j.begin(); it_j != bucket_j.end(); it_j++ ) {
			for ( it_k = bucket_k.begin(); it_k != bucket_k.end(); it_k++ ) {

				//FIXME Talvez de pra usar informacoes do suporte para fazer este teste
				//if ( (it_i->first == it_j->first) || (it_i->first == it_k->first) || (it_j->first == it_k->first) ) {
				//	continue;
				//}

				MUX( (*it_i), (*it_j), (*it_k) );
				if ( this->stop ) return;
				MUX( (*it_i), (*it_k), (*it_j) );
				if ( this->stop ) return;
				MUX( (*it_j), (*it_i), (*it_k) );
				if ( this->stop ) return;
				MUX( (*it_j), (*it_k), (*it_i) );
				if ( this->stop ) return;
				MUX( (*it_k), (*it_j), (*it_i) );
				if ( this->stop ) return;
				MUX( (*it_k), (*it_i), (*it_j) );
				if ( this->stop ) return;
			}
		}
	}
}

void EquationOptimizer::AND( const BucketElement & e1, const BucketElement & e2 ) {

	if ( !this->runningLookAhead ) {
		if ( this->iBucket <= this->nVars ) {
			if ( (e1.support & e2.support) != 0 ) {
				return;
			}
		}
		else {
			unsigned int sharedSupport = (e1.support & e2.support);
			int sharedLits = Functional::oneCounter( sharedSupport );
			if ( sharedLits > (this->iBucket - this->nVars) ) {
				return;
			}
		}
	}

	BucketElement er;
	er.function = this->functionPool.getMemory();
	Functional::AND( er.function, e1.function, e2.function, this->nWords );
	er.support = e1.support | e2.support;
	er.equation = e1.equation * e2.equation;
	er.ccDist = 1 + std::max( e1.ccDist, e2.ccDist );

	if ( abandon( er ) == false ) {
		insertInBuckets( er );
	}
	else {
		this->functionPool.giveBackMemory();
	}
}

void EquationOptimizer::OR( const BucketElement & e1, const BucketElement & e2 ) {

	if ( !this->runningLookAhead ) {
		if ( this->iBucket <= this->nVars ) {
			if ( (e1.support & e2.support) != 0 ) {
				return;
			}
		}
		else {
			unsigned int sharedSupport = (e1.support & e2.support);
			int sharedLits = Functional::oneCounter( sharedSupport );
			if ( sharedLits > (this->iBucket - this->nVars) ) {
				return;
			}
		}
	}

	BucketElement er;
	er.function = this->functionPool.getMemory();
	Functional::OR( er.function, e1.function, e2.function, this->nWords );
	er.support = e1.support | e2.support;
	er.equation = e1.equation + e2.equation;
	er.ccDist = 1 + std::max( e1.ccDist, e2.ccDist );

	if ( abandon( er ) == false ) {
		insertInBuckets( er );
	}
	else {
		this->functionPool.giveBackMemory();
	}
}

void EquationOptimizer::XOR( const BucketElement & e1, const BucketElement & e2 ) {

	if ( !this->runningLookAhead ) {
		if ( this->iBucket <= this->nVars ) {
			if ( (e1.support & e2.support) != 0 ) {
				return;
			}
		}
		else {
			unsigned int sharedSupport = (e1.support & e2.support);
			int sharedLits = Functional::oneCounter( sharedSupport );
			if ( sharedLits > (this->iBucket - this->nVars) ) {
				return;
			}
		}
	}

	BucketElement er;
	er.function = this->functionPool.getMemory();
	Functional::XOR( er.function, e1.function, e2.function, this->nWords );
	er.support = e1.support | e2.support;
	er.equation = e1.equation ^ e2.equation;
	er.ccDist = 1 + std::max( e1.ccDist, e2.ccDist );

	if ( abandon( er ) == false ) {
		insertInBuckets( er );
	}
	else {
		this->functionPool.giveBackMemory();
	}
}

void EquationOptimizer::MUX( const BucketElement & e1, const BucketElement & e2, const BucketElement & e3 ) {

	BucketElement er;
	er.function = this->functionPool.getMemory();
	Functional::MUX( er.function, e1.function, e2.function, e3.function, this->nWords );
	er.support = e1.support | e2.support | e3.support;
	er.equation = EquationData::mux( e1.equation, e2.equation, e3.equation );
	er.ccDist = 1 + std::max( e1.ccDist, std::max( e2.ccDist, e3.ccDist ) );

	if ( abandon( er ) == false ) {
		insertInBuckets( er );
	}
	else {
		this->functionPool.giveBackMemory();
	}
}


bool EquationOptimizer::abandon( BucketElement & element ) {

	auto itCCcomb = this->cubeCofactorsComb.find( element.function );

	if ( itCCcomb == this->cubeCofactorsComb.end() ) { // If f is not allowed, then abandon
		return true;
	}
	else {
		auto itCC =  this->cubeCofactors.find( element.function );
		if ( itCC == this->cubeCofactors.end() ) {
			element.ccDist = 1;
		}
		else {
			element.ccDist = 0;
		}
		return false;
	}
}

void EquationOptimizer::insertInBuckets( BucketElement & element ) {

	if ( this->runningLookAhead ) {
		if ( Functional::equals( this->targetFunction, element.function, this->nWords ) ) {
			if ( ( this->lookAheadSolution.equation.getLiterals() == 0 ) || ( this->lookAheadSolution.equation.getLiterals() > element.equation.getLiterals() ) ) {
				this->lookAheadSolution = element;
			}
		}
		return;
	}

	auto status = this->alreadyVisited.insert( element.function );

	if ( (status.second == true) || ( Functional::equals( this->targetFunction, element.function, this->nWords ) ) ) { // If f was not visited yet, then insert in the respective bucket

		if ( this->verbose ) {
			std::cout << Functional::toHex( element.function, this->nWords ) << " = ";
			std::cout << element.equation.getEquation() << " -> "; 
			std::cout << Functional::supportToBin( element.support ) << std::endl;
		}

		int lits = element.equation.getLiterals();
		Functional::Order order = Functional::order( element.function, this->targetFunction, this->nWords );

		if ( order == Functional::Order::NOTCOMPARABLE ) {
			this->notComparableBuckets[ lits ].push_back( element );
		}

		if ( order == Functional::Order::SMALLER ) {
			this->smallerBuckets[ lits ].push_back( element );
		}

		if ( order == Functional::Order::LARGER ) {
			this->largerBuckets[ lits ].push_back( element );
		}

		if ( order == Functional::Order::EQUAL ) {
			registerSolution( element.equation );
			if ( this->solutions.size() == this->desiredNumberOfSolutions ) {
				this->stop = true;
			}
		}
	}
}

/* Inser and keep the vector sorted by literal count of each solution */
void EquationOptimizer::registerSolution( EquationData & newSolution ) {

	int index = this->solutions.size();

	for ( int i = this->solutions.size()-1; i >= 0; i-- ) {
		if ( newSolution == this->solutions[i] ) {
			return;
		}
		
		if ( newSolution.getLiterals() < this->solutions[i].getLiterals() ) {
			index = i;
		}
	}

	if ( index < this->solutions.size() ) {
		this->solutions.push_back( this->solutions[index] );
		this->solutions[ index ] = newSolution;
	}
	else {
		this->solutions.push_back( newSolution );
	}
}

EquationDataVector & EquationOptimizer::getSolutions() {
	return this->solutions;
}

void EquationOptimizer::printSolutions() {
	for ( auto equationData : this->solutions ) {
		std::cout << equationData.getEquation() << ";" << equationData.getLiterals() << std::endl;
	}
}

void EquationOptimizer::printBucket( Bucket & bucket, std::string label ) {

	std::cout << "\t" << label << std::endl;
	for ( auto element : bucket ) {
		//std::cout << "\t" << element.equation.getEquation() << std::endl;
		//std::cout << "\t" << element.equation.getEquation() << " " << Functional::getHammingDist( this->targetFunction, element.function, nWords ) << std::endl;
		std::cout << "\t" << element.equation.getEquation() << " | " << element.ccDist << " | " << Functional::getHammingDist( this->targetFunction, element.function, this->nWords ) << " | ";
		std::cout << Functional::toHex( element.function, this->nWords ) << std::endl;

	}
	if ( bucket.empty() ) {
		std::cout << "\tEMPTY" << std::endl;
	}
	std::cout << std::endl;
}

void EquationOptimizer::printLiteralDisapearanceMatrix() {

	for ( int iVar = 0; iVar < this->nVars; iVar++ ) {
		for ( int jVar = 0; jVar < this->nVars; jVar++ ) {
			std::cout << "(" << this->literalDisapearanceMatrix[ iVar ][ jVar ].negCof_negLit << ",";
			std::cout << this->literalDisapearanceMatrix[ iVar ][ jVar ].negCof_posLit << ",";
			std::cout << this->literalDisapearanceMatrix[ iVar ][ jVar ].posCof_negLit << ",";
			std::cout << this->literalDisapearanceMatrix[ iVar ][ jVar ].posCof_posLit << ")  ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int EquationOptimizer::getIBucket() {
	return this->iBucket;
}

long int EquationOptimizer::getNCC() {
	return this->nCC;
}

long int EquationOptimizer::getNCCcomb() {
	return this->nCCcomb;
}

double EquationOptimizer::getKillGraphRuntime() {
	return this->killGraphRuntime;
}

double EquationOptimizer::getCubeCofactorRuntime() {
	return this->cubeCofactorRuntime;
}

double EquationOptimizer::getCubeCofactorCombRuntime() {
	return this->cubeCofactorCombRuntime;
}

double EquationOptimizer::getBucketCombRuntime() {
	return this->bucketCombRuntime;
}

} //namespace Factoring
