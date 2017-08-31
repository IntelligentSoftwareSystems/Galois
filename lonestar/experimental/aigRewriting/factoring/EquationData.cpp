#include "../factoring/EquationData.h"

#include <algorithm>
#include <iostream>

namespace Factoring {

EquationData::EquationData() {
	this->equation = "";
	this->series = 0;
	this->parallel = 0;
	this->literals = 0;
}

EquationData::EquationData( String equation, short series, short parallel, short literals ) {
	
	this->equation = equation;
	this->series = series;
	this->parallel = parallel;
	this->literals = literals;
}

EquationData & EquationData::operator=( const EquationData & rhs ) {

	this->equation = rhs.equation;
	this->series = rhs.series;
	this->parallel = rhs.parallel;
	this->literals = rhs.literals;

	return *this;
}

bool EquationData::operator==( const EquationData & rhs ) const {

	if( this->equation == rhs.equation ) {
		return true;
	}
	else {
		return false;
	}
}

EquationData EquationData::operator+( const EquationData & rhs ) const {

	String equation = "(" + this->equation + "+" + rhs.getEquation() + ")";
	short s = std::max( this->series, rhs.series );
	short p = this->parallel + rhs.parallel;
	short l = this->literals + rhs.literals;

	return EquationData( equation, s, p, l );
}

EquationData EquationData::operator*( const EquationData & rhs ) const {

	String equation = "(" + this->equation + "*" + rhs.getEquation() + ")";
	short s = this->series + rhs.series;
	short p = std::max( this->parallel, rhs.parallel );
	short l = this->literals + rhs.literals;
	
	return EquationData( equation, s, p, l );
}

EquationData EquationData::operator^( const EquationData & rhs ) const {

	String equation = "(" + this->equation + "^" + rhs.getEquation() + ")";
	short s = this->series + rhs.series;
	short p = this->parallel + rhs.parallel;
	short l = this->literals + rhs.literals;
	
	return EquationData( equation, s, p, l );
}

EquationData EquationData::mux( const EquationData ed1, const EquationData ed2, const EquationData ed3 ) {

	String equation = "m((" + ed1.getEquation() + "),(" + ed2.getEquation() + "),(" + ed3.getEquation() + "))";
	short s = std::max( ed1.getSeries(), ed2.getSeries() );
	s = std::max( s, ed3.getSeries() );
	short p = ed1.getParallel() + ed2.getParallel() + ed3.getParallel();
	short l = ed1.getLiterals() + ed2.getLiterals() + ed3.getLiterals();

	return EquationData( equation, s, p, l );
}

void EquationData::appendParallelLiteral( String & literal ) {
	this->equation = "(" + this->equation + "+" + literal + ")";
	this->literals++;
	this->parallel++;
}

void EquationData::appendSeriesLiteral( String & literal ) {
	this->equation = "(" + this->equation + "*" + literal + ")";
	this->literals++;
	this->series++;
}

String EquationData::getEquation() const {
	return this->equation;
}

short EquationData::getSeries() const {
	return this->series;
}

short EquationData::getParallel() const {
	return this->parallel;
}

short EquationData::getLiterals() const {
	return this->literals;
}

} // namespace Factoring
