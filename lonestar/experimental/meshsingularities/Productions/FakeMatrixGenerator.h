/*
 * FakeMatrixGenerator.h
 *
 *  Created on: Aug 28, 2013
 *      Author: kjopek
 */

#ifndef FAKEMATRIXGENERATOR_H_
#define FAKEMATRIXGENERATOR_H_

#include "MatrixGeneration/GenericMatrixGenerator.hxx"

/*
 * TODO: implement all of classes below with appropriate content
 *
 * current implementation is only a fake for test purposes.
 */

class FakeMatrixGenerator : public GenericMatrixGenerator
{
public:
	virtual std::vector<EquationSystem*>* CreateMatrixAndRhs(TaskDescription& task_description);
	virtual void checkSolution(std::map<int,double> *solution_map, double (*f)(int dim, ...));
	virtual std::vector<int>* GetProductionParameters(int polynomial_degree){
                std::vector<int> *parameters = new std::vector<int>(4);
                (*parameters)[0] = this->getiSize(polynomial_degree);
                (*parameters)[1] = this->getLeafSize(polynomial_degree);
                (*parameters)[2] = this->getA1Size(polynomial_degree);
                (*parameters)[3] = this->getANSize(polynomial_degree);
                return parameters;
        }

	virtual int getiSize(int p) {
		return 24*(p-1)*(p-1) + 48*(p-1) + 26;
	}

	virtual int getLeafSize(int p) {
		return 56*(p-1)*(p-1)*(p-1) + 108*(p-1)*(p-1) + 54*(p-1);
	}

	virtual int getA1Size(int p) {
		return FakeMatrixGenerator::getLeafSize(p) + FakeMatrixGenerator::getiSize(p) + 96*(p-1)*(p-1) + 192*(p-1) + 98;
	}

	virtual int getANSize(int p) {
		return FakeMatrixGenerator::getLeafSize(p) + FakeMatrixGenerator::getiSize(p) + 8*(p-1)*(p-1)*(p-1) + 36*(p-1)*(p-1) + 54*(p-1) + 27;
	}

};

class Point2DMatrixGenerator : public FakeMatrixGenerator
{
public:

	virtual int getiSize(int p) {
		return 2*(p-1)+3;
	}

	virtual int getLeafSize(int p) {
		return 3*(p-1)*(p-1) + 8*(p-1)+6;
	}

	virtual int getA1Size(int p) {
		return 3*(p-1)*(p-1) + 10*(p-1) + 8;
	}

	virtual int getANSize(int p) {
		return 3*(p-1)*(p-1) + 8*(p-1) + 6;
	}

};

class Edge2DMatrixGenerator : public FakeMatrixGenerator
{
public:

	virtual int getiSize(int p) {
		return (p-1)+2;
	}

	virtual int getLeafSize(int p) {
		return (p-1)*(p-1) + 4*(p-1) + 4;
	}

	virtual int getA1Size(int p) {
		return (p-1)*(p-1) + 4*(p-1) + 4;
	}

	virtual int getANSize(int p) {
		return (p-1)*(p-1) + 4*(p-1) + 4;
	}

};

class PointCentral2DMatrixGenerator : public FakeMatrixGenerator
{
public:
	virtual int getiSize(int p) {
		return 8*(p-1)+8;
	}

	virtual int getLeafSize(int p) {
		return getA1Size(p) - 8*p;
	}

	virtual int getA1Size(int p) {
		return 12*(p-1)*(p-1) + 36*(p-1) + 24;
	}

	virtual int getANSize(int p) {
		return 16*(p-1)*(p-1) + 32*(p-1) + 17;
	}

};


class Point3DMatrixGenerator : public FakeMatrixGenerator
{
public:

	virtual int getiSize(int p) {
		return 3*(p-1)*(p-1) + 9*(p-1) + 7;
	}

	virtual int getLeafSize(int p) {
		return 7*(p-1)*(p-1)*(p-1) + 12*(p-1)*(p-1) + 42*(p-1) + 23;
	}

	virtual int getA1Size(int p) {
		return 8*(p-1)*(p-1)*(p-1) + 33*(p-1)*(p-1) + 51*(p-1) + 26;
	}

	virtual int getANSize(int p) {
		return 8*(p-1)*(p-1)*(p-1) + 27*(p-1)*(p-1) + 45*(p-1) + 24;
	}

};

class Edge3DMatrixGenerator : public FakeMatrixGenerator
{
public:

	virtual int getiSize(int p) {
		return 2*(p-1)*(p-1) + 5*(p-1) + 6;
	}

	virtual int getLeafSize(int p) {
		return getA1Size(p) - 2*(p-1)*(p-1) - 2*(p-1) - 4;
	}

	virtual int getA1Size(int p) {
		return 3*(p-1)*(p-1)*(p-1) + 14*(p-1)*(p-1) + 28*(p-1) + 16;
	}

	virtual int getANSize(int p) {
		return 4*(p-1)*(p-1)*(p-1) + 18*(p-1)*(p-1) + 30*(p-1) + 12;
	}

};

class Face3DMatrixGenerator : public FakeMatrixGenerator
{
public:

	virtual int getiSize(int p) {
		return (p-1)*(p-1) + 4*(p-1) + 4;
	}

	virtual int getLeafSize(int p) {
		return getA1Size(p);
	}

	virtual int getA1Size(int p) {
		return (p-1)*(p-1)*(p-1) + 6*(p-1)*(p-1) + 12*(p-1) + 8;
	}

	virtual int getANSize(int p) {
		return getA1Size(p);
	}

};


class PointCentral3DMatrixGenerator : public FakeMatrixGenerator
{
public:

	virtual int getiSize(int p) {
		return 4*(3*(p-1)*(p-1) + 9*(p-1) + 7)-4*(p-1)-8;
	}

	virtual int getLeafSize(int p) {
		return 4*(7*(p-1)*(p-1)*(p-1) + 12*(p-1)*(p-1) + 42*(p-1) + 23) - 4*(p-1)*(p-1)-16*(p-1)-16;
	}

	virtual int getA1Size(int p) {
		return 4*(8*(p-1)*(p-1)*(p-1) + 33*(p-1)*(p-1) + 51*(p-1) + 26) - 4*(p-1)*(p-1)-16*(p-1) - 16;
	}

	virtual int getANSize(int p) {
		return 4*(8*(p-1)*(p-1)*(p-1) + 27*(p-1)*(p-1) + 45*(p-1) + 24) - 4*(p-1)*(p-1)-16*(p-1) - 16;
	}

};

class Anisotropic3DMatrixGenerator : public FakeMatrixGenerator
{
public:

	virtual int getiSize(int p) {
		return 4*this->getiSize(p)-4*(p-1)-8;
	}

	virtual int getLeafSize(int p) {
		return 8*this->getLeafSize(p) - 36*(p-1)*(p-1)-120*(p-1)-8;
	}

	virtual int getA1Size(int p) {
		return 8*this->getA1Size(p) - 36*(p-1)*(p-1)-120*(p-1) - 8;
	}

	virtual int getANSize(int p) {
		return 8*this->getANSize(p) - 36*(p-1)*(p-1)-120*(p-1) - 8;
	}

};


class FakeMatrixGeneratorFaceAniso : public FakeMatrixGenerator
{
public:
	virtual int getiSize(int p) {
		return 3*(p-1)*(p-1);
	}

	virtual int getLeafSize(int p) {
		return 9*(p-1)*(p-1) + 3*(p-1)*(p-1)*(p-1);
	}

	virtual int getA1Size(int p) {
		return 6*FakeMatrixGeneratorFaceAniso::getLeafSize(p);
	}

	virtual int getANSize(int p) {
		return 6*FakeMatrixGeneratorFaceAniso::getLeafSize(p);
	}

};


#endif /* FAKEMATRIXGENERATOR_H_ */
