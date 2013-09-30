#ifndef IFUNCTION_HXX_
#define IFUNCTION_HXX_

class IFunction
{
	protected:
		double* coordinates;
		bool* neighbours;


	public:
		IFunction(double* coordinates, bool* neighbours) :
			coordinates(coordinates), neighbours(neighbours)
		{

		}
		IFunction()
		{

		}

		virtual ~IFunction()
		{

		}
};
#endif
