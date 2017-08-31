#ifndef EQUATION_DATA_H
#define EQUATION_DATA_H

#include <string>
#include <vector>

namespace Factoring {

typedef std::string String;

class EquationData {

      String equation;
      short series;
      short parallel;
      short literals;

public:

      EquationData();
      EquationData( String equation, short series, short parallel, short literals );
      
      EquationData & operator=( const EquationData & rhs );
      bool operator==( const EquationData & rhs ) const;
      EquationData operator+( const EquationData & rhs ) const;
      EquationData operator*( const EquationData & rhs ) const;
      EquationData operator^( const EquationData & rhs ) const;
      EquationData operator~() const;
      static EquationData mux( const EquationData ed1, const EquationData ed2, const EquationData ed3 ); // MUX
      void appendSeriesLiteral( String & literal );
      void appendParallelLiteral( String & literal );

      String getEquation() const;
      short getSeries() const;
      short getParallel() const;
      short getLiterals() const;
};

} // namespace Factoring

#endif
